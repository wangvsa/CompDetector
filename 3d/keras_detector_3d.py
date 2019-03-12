from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv3D, Flatten, Activation, MaxPooling3D, BatchNormalization
from keras.utils import multi_gpu_model
import keras
import keras.backend as K
import numpy as np
import sys, glob, argparse, random
from create_dataset import get_flip_error, split_to_blocks, hdf5_to_numpy


NX, NY, NZ = 16, 16, 16
WINDOWS_PER_IMAGE = 64*64*64/NX/NY/NZ
#variables = ["dens", "pres", "temp"]
variables = ["dens"]
EPOCHS = 2

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

class FlashDatasetGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, zero_propagation=True):
        self.zero_propagation = zero_propagation
        clean_files = glob.glob(data_dir+"/*/clean/*")
        # if zero_propagation, create 0-propagation error dataset at runtime
        if zero_propagation:
            error_files = list(clean_files)
        else:
            error_files = glob.glob(data_dir+"/*/error/*")
        files = clean_files + error_files
        labels = np.append(np.zeros(len(clean_files)), np.ones(len(error_files)))
        print "clean files:", len(clean_files), ", error files:", len(error_files)
        self.batch_size = batch_size

        self.files, self.labels = shuffle_two_lists(files, labels)


    def __len__(self):
        return np.floor(len(self.files) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_y = self.labels[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_x = []
        for file_index in range(idx*self.batch_size, (idx+1)*self.batch_size):
            img = np.load(self.files[file_index])
            if img.ndim == 3: img = np.expand_dims(img, axis=-1)
            # if zero_propagation, insert an error to create the error data at runtime
            if self.zero_propagation and self.labels[file_index]:
                # Insert an error
                x, y, z, v = random.randint(4, img.shape[0]-3), random.randint(4, img.shape[1]-3),\
                            random.randint(4, img.shape[2]-3), random.randint(0, img.shape[3]-1)
                error = get_flip_error(img[x,y,z,v], 15)
                img[x, y, z, v] = error
            batch_x.append(calc_gradient(img))
        return np.array(batch_x), batch_y

    def get_true_labels(self):
        truth = [0] * (self.clean_labels/WINDOWS_PER_IMAGE) + [1] * (self.error_labels/WINDOWS_PER_IMAGE)
        return np.array(truth)

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

try:
    model = multi_gpu_model(model)
except:
    pass

#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3, amsgrad=False)
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

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
    recall, fpr = tp / error_samples, fp / total
    accuracy = (tp + tn) / total
    print 'TP: %s (%i/%i), FP: %s (%i/%i)' %(recall, tp, error_samples, fpr, fp, total)
    print 'ACC: %s, TN: %i, FN: %i' %(accuracy, tn, fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dataset", help="The path to the splitted training set")
    parser.add_argument("-e", "--test_dataset", help="The path to the splitted test set")
    parser.add_argument("-n", "--epochs", help="How many epochs for training")
    parser.add_argument("-m", "--model", help="Specify thet model file")
    parser.add_argument("-d", "--detect_dataset", help="Run the detector on a unsplitted dataset")
    args = parser.parse_args()

    model_file = "./models/model_keras.h5"
    if args.model:
        model_file = args.model
    if args.train_dataset:
        if args.epochs: EPOCHS = int(args.epochs)
        data_gen = FlashDatasetGenerator(args.train_dataset, 64, zero_propagation=True)
        #model.load_weights(model_file)
        model.fit_generator(generator=data_gen, use_multiprocessing=True, workers=8, epochs=EPOCHS, shuffle=True)
        model.save_weights('model_keras.h5')
        print model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8, verbose=1)
    elif args.test_dataset:
        data_gen = FlashDatasetGenerator(args.test_dataset, 64, zero_propagation=True)
        model.load_weights(model_file)
        #print model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8, verbose=1)
        scores = model.predict_generator(generator=data_gen, use_multiprocessing=False, workers=1, verbose=1)
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
            print "Threshold =", threshold
            pred = (scores >= threshold)    # shape of (N, WINDOWS_PER_IMAGE)
            #pred = np.any(pred, axis=1)
            compute_metrics(pred, data_gen.labels)
    elif args.detect_dataset:
        model.load_weights(model_file)
        error, nan_count = 0, 0
        files = list(glob.glob(args.detect_dataset+"/*chk_*"))
        files.sort()
        for filename in files:
            dens = hdf5_to_numpy(filename)
            if np.isnan(dens).any():
                error, nan_count = error+1, nan_count+1
                continue
            dens_blocks = np.expand_dims(np.squeeze(split_to_blocks(dens)), -1)
            for i in range(dens_blocks.shape[0]):
                dens_blocks[i] = calc_gradient(dens_blocks[i])
            pred = model.predict(dens_blocks) > 0.5
            error += np.any(pred)
            if np.any(pred) == 0:
                print filename
        print "detected %s error samples (%s nan samples), total: %s, recall: %s" %(error, nan_count, len(files), error*1.0/len(files))
