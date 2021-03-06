from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv3D, Flatten, Activation, MaxPooling3D, BatchNormalization
from keras.utils import multi_gpu_model
import keras
import keras.backend as K
import numpy as np
import sys, glob, argparse, random
import warnings
from timeit import default_timer as timer
from create_dataset import get_flip_error, split_to_blocks, split_to_windows, hdf5_to_numpy, read_data


NX, NY, NZ = 16, 16, 16
WINDOWS_PER_IMAGE = 64*64*64/NX/NY/NZ
#variables = ["dens", "pres", "temp"]
variables = ["dens"]
BATCH_SIZE = 64


def calc_gradient(data):
    d = data[:,:,:,0]
    grad = np.gradient(d)
    if len(grad) == 3:      # 3D data
        d = np.sqrt(grad[0]**2  + grad[1]**2 + grad[2]**2)
        return d.reshape(data.shape)

# shuffle two list at the same order
def shuffle_two_lists(l1, l2):
    l = list(zip(l1, l2))
    random.shuffle(l)
    r1, r2 = zip(*l)
    return list(r1), list(r2)


# Those are errors so easy to detect
# Do not need to run the neural network
def simple_pre_detection(d):
    if np.isnan(d).any():
        return True
    if (abs(d) > 10e4).any():
        return True
    return False


# The Data Generator is only used for training and testing on splitted dataset
class FlashDatasetGenerator(keras.utils.Sequence):
    def preprocess(self, tmp_clean_files, tmp_error_files):
        clean_files = []
        error_files = []
        for f in tmp_clean_files:
            if not simple_pre_detection(np.load(f)):
                clean_files.append(f)
        for f in tmp_error_files:
            if not simple_pre_detection(np.load(f)):
                error_files.append(f)
        self.class_weight = {0:len(clean_files), 1:len(error_files)}
        return clean_files, error_files

    def __init__(self, data_dir, batch_size, zero_propagation=True, bits=15):
        t1 = timer()
        self.bits = bits
        self.zero_propagation = zero_propagation
        clean_files = glob.glob(data_dir+"/*/clean/*")
        if zero_propagation: # insert error at runtime
            error_files = list(clean_files)
        else:
            error_files = glob.glob(data_dir+"/*/error/*")
        print("clean files:", len(clean_files), ", error files:", len(error_files))
        clean_files, error_files = self.preprocess(clean_files, error_files)
        files = clean_files + error_files
        labels = np.append(np.zeros(len(clean_files)), np.ones(len(error_files)))
        print("clean files:", len(clean_files), ", error files:", len(error_files))
        self.batch_size = batch_size
        self.files, self.labels = shuffle_two_lists(files, labels)
        t2 = timer()
        print("init dataset time:", (t2-t1))

    def __len__(self):
        return np.ceil(len(self.files) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_y = self.labels[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_x = []
        for file_index in range(idx*self.batch_size, min((idx+1)*self.batch_size, len(self.files))):
            img = np.load(self.files[file_index])
            if img.ndim == 3: img = np.expand_dims(img, axis=-1)
            # if zero_propagation, insert an error to create the error data at runtime
            if self.zero_propagation and self.labels[file_index]:
                # Insert an error
                x, y, z, v = random.randint(4, img.shape[0]-3), random.randint(4, img.shape[1]-3),\
                            random.randint(4, img.shape[2]-3), random.randint(0, img.shape[3]-1)
                error, bit = get_flip_error(img[x,y,z,v], 0, self.bits)
                img[x, y, z, v] = error
            batch_x.append(calc_gradient(img))
        return np.array(batch_x), batch_y

model = Sequential([
    Conv3D(32, (3,3,3), input_shape=(NX, NY, NZ, len(variables))),
    #BatchNormalization(axis=-1),
    Activation('relu'),
    Conv3D(32, (3,3,3)),
    #BatchNormalization(axis=-1),
    Activation('relu'),
    Conv3D(32, (3,3,3)),
    #BatchNormalization(axis=-1),
    Activation('relu'),
    MaxPooling3D(pool_size=(3, 3, 3)),
    Flatten(),
    Dropout(0.25),
    Dense(1, activation='sigmoid')
])


def compute_metrics(pred_labels, true_labels):
    true_labels = np.array(true_labels).reshape(pred_labels.shape)
    error_samples = np.sum(true_labels) * 1.0
    total = len(true_labels) * 1.0
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    recall, fpr = tp / error_samples, fp / (total-error_samples)
    accuracy = (tp + tn) / total
    print('TP: %s (%i/%i), FP: %s (%i/%i)' %(recall, tp, error_samples, fpr, fp, total-error_samples))
    print('ACC: %s, TN: %i, FN: %i' %(accuracy, tn, fn))


def get_parsed_arguments():
    parser = argparse.ArgumentParser()

    # Rquired action, need to choose from one of the following.
    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument("--train", help="Train the model with splitted dataset", action="store_true")
    parser_group.add_argument("--test", help="Test the model with splitted dataset", action="store_true")
    parser_group.add_argument("--detect", help="Run the detector (trained model) on a unsplitted dataset")

    # optional
    parser.add_argument("-n", "--epochs", help="How many epochs for training", type=int, default=3)
    parser.add_argument("-m", "--model", help="Specify the model file", type=str, default="./models/model_keras.h5")
    parser.add_argument("-k", help="the number of k", type=int, default=0)
    parser.add_argument("-b", "--bits", help="Error bits range", type=int, default=15)
    parser.add_argument("--data", help="Path to splitted dataset", default="")

    return parser.parse_args()


def detection(model, windows):
    if simple_pre_detection(windows):
        return True
    # consider all warnings as errors
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            for i in range(windows.shape[0]):
                windows[i] = calc_gradient(windows[i])
            pred = model.predict(windows) > 0.5
            hasError = np.any(pred)
        except Warning as e:
            print("warnning captured:", e)
            hasError = True
    return hasError


if __name__ == "__main__":
    # Construct the model
    try: model = multi_gpu_model(model)
    except: pass
    #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3, amsgrad=False)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    args = get_parsed_arguments()
    model_file = args.model

    if args.train:
        data_gen = FlashDatasetGenerator(args.data, BATCH_SIZE, args.k==0, args.bits)
        #model.load_weights(model_file)
        t1 = timer()
        model.fit_generator(generator=data_gen, use_multiprocessing=True, class_weight=data_gen.class_weight,
                            workers=8, epochs=args.epochs, shuffle=True)
        model.save_weights(model_file)
        t2 = timer()
        print("training time:", (t2-t1))
        print(model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8, verbose=1))
    elif args.test:
        data_gen = FlashDatasetGenerator(args.data, BATCH_SIZE, args.k==0, args.bits)
        model.load_weights(model_file)
        scores = model.predict_generator(generator=data_gen, use_multiprocessing=False, workers=1, verbose=1)
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
            print("Threshold =", threshold)
            pred = (scores >= threshold)
            compute_metrics(pred, data_gen.labels)
    elif args.detect:
        model.load_weights(model_file)
        error = 0
        # match both for clean checkpoint files and k-delay corrupted files
        files = glob.glob(args.detect+"/*chk_*")+glob.glob(args.detect+"/*error*")
        files.sort()
        files = files[0:200]
        for filename in files:
            dens = read_data(filename)
            dens_blocks = np.expand_dims(np.squeeze(split_to_windows(dens)), -1)
            if detection(model, dens_blocks):
                error += 1
        print("detected %s error samples, total: %s, recall: %s" %(error, len(files), error*1.0/len(files)))
