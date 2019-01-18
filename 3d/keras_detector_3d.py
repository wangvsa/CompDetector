from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv3D, Flatten, Activation, MaxPooling3D, BatchNormalization
from keras.utils import multi_gpu_model
import keras
import keras.backend as K
import numpy as np
import sys, glob, argparse, random
from create_dataset import get_flip_error


NX, NY, NZ = 16, 16, 16
WINDOWS_PER_IMAGE = 128*128*128/NX/NY/NZ
#variables = ["dens", "pres", "temp"]
variables = ["dens"]
EPOCHS = 3

class FlashDatasetGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, zero_propagation=True, testing=False):
        self.zero_propagation = zero_propagation
        clean_files = glob.glob(data_dir+"/*/clean/*")
        # if zero_propagation, create 0-propagation error dataset at runtime
        if zero_propagation:
            error_files = list(clean_files)
        else:
            error_files = glob.glob(data_dir+"/*/error/*")
        self.files = clean_files + error_files
        self.clean_labels, self.error_labels = len(clean_files), len(error_files)
        print "clean files:", self.clean_labels, ", error files:", self.error_labels
        self.labels = np.append(np.zeros(self.clean_labels), np.ones(self.error_labels))
        self.batch_size = batch_size

        # When testing, first determine which window has error
        # only one window in an image has an error
        self.testing = testing
        self.error_blocks = np.random.randint(WINDOWS_PER_IMAGE, size=self.error_labels/WINDOWS_PER_IMAGE)

        # keep errors and original values
        self.error_information = []

    def __len__(self):
        return np.ceil(len(self.files) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_y = self.labels[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_x = []
        for file_index in range(idx*self.batch_size, (idx+1)*self.batch_size):
            img = np.load(self.files[file_index])
            #img = img.reshape((NX,NY,NZ,1))
            #img = np.fromfile(self.files[file_index], dtype=np.float64).reshape((NX,NY,NZ,3))[:,:,:,0:1]
            # if zero_propagation, insert an error to create the error data at runtime
            if self.zero_propagation:
                if self.testing :   # testing, only one block in an image has an error
                    timestep = (file_index - self.clean_labels) / WINDOWS_PER_IMAGE
                    if (file_index >= self.clean_labels) and (file_index % WINDOWS_PER_IMAGE == self.error_blocks[timestep]):
                        # Insert an error
                        x, y, z, v = random.randint(0, img.shape[0]-1), random.randint(0, img.shape[1]-1),\
                                    random.randint(0, img.shape[2]-1), random.randint(0, img.shape[3]-1)
                        old = img[x, y, z, v]
                        img[x, y, z, v] = get_flip_error(img[x, y, z, v], 5)

                        # record errors
                        self.error_information.append([old, img[x,y,z,v]])
                else:               # training, then all blocks in an image has an error
                    if file_index >= self.clean_labels:
                        # Insert an error
                        x, y, z, v = random.randint(1, img.shape[0]-2), random.randint(1, img.shape[1]-2),\
                                    random.randint(1, img.shape[2]-2), random.randint(0, img.shape[3]-1)
                        img[x, y, z, v] = get_flip_error(img[x, y, z, v], 5)
            # Normalization
            #img_min = np.min(img)
            #img_max = np.max(img)
            #if img_max != img_min:
            #    img = (img - img_min) / (img_max - img_min)
            #img = img + 0.1
            #img = np.log(img)
            batch_x.append(img)
        return np.array(batch_x), batch_y

    def get_true_labels(self):
        truth = [0] * (self.clean_labels/WINDOWS_PER_IMAGE) + [1] * (self.error_labels/WINDOWS_PER_IMAGE)
        return np.array(truth)

model = Sequential([
    Conv3D(32, (3,3,3), input_shape=(NX, NY, NZ, len(variables))),
    Activation('relu'),
    #BatchNormalization(),
    #MaxPooling3D(pool_size=(3, 3, 3)),
    Conv3D(32, (3,3,3), activation='relu'),
    #MaxPooling3D(pool_size=(3, 3, 1)),
    Conv3D(32, (3,3,3), activation='relu'),
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
model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])

def compute_metrics(pred_labels, true_labels):
    clean_samples = np.sum(true_labels == 0) * 1.0
    error_samples = np.sum(true_labels == 1) * 1.0
    total = clean_samples + error_samples * 1.0
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp_index = np.nonzero(np.logical_and(pred_labels == 1, true_labels == 1))[0]
    tp = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn_index = np.nonzero(np.logical_and(pred_labels == 0, true_labels == 1))[0]
    fn = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    recall, fpr = tp / error_samples, fp / total
    accuracy = (tp + tn) / total
    print 'TP: %s (%i/%i), FP: %s (%i/%i)' %(recall, tp, error_samples, fpr, fp, total)
    print 'ACC: %s, TN: %i, FN: %i' %(accuracy, tn, fn)
    return tp_index, fn_index

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dataset", help="The path to the training set")
    parser.add_argument("-e", "--test_dataset", help="The path to the test set")
    parser.add_argument("-n", "--epochs", help="How many epochs for training")
    args = parser.parse_args()

    if args.train_dataset:
        if args.epochs: EPOCHS = int(args.epochs)
        data_gen = FlashDatasetGenerator(args.train_dataset, 64, zero_propagation=True, testing=False)
        model.load_weights('model_keras.h5')
        model.fit_generator(generator=data_gen, use_multiprocessing=True, workers=8, epochs=EPOCHS, shuffle=True)
        model.save_weights('model_keras.h5')
        print model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8, verbose=1)
    elif args.test_dataset:
        data_gen = FlashDatasetGenerator(args.test_dataset, 64, zero_propagation=True, testing=True)
        model.load_weights('model_keras.h5')
        print model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8, verbose=1)
        scores = model.predict_generator(generator=data_gen, use_multiprocessing=False, workers=1, verbose=1)
        scores = scores.reshape((len(scores)/WINDOWS_PER_IMAGE, WINDOWS_PER_IMAGE))
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
            print "Threshold =", threshold
            pred = (scores >= threshold)    # shape of (N, WINDOWS_PER_IMAGE)
            pred = np.any(pred, axis=1)
            tp_index, fn_index = compute_metrics(pred, data_gen.get_true_labels())
            #print np.array(data_gen.error_information)[fn_index-50]
