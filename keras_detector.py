from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
import keras
import numpy as np
import sys, glob

NX, NY = 60, 60

class FlashDatasetGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size):
        clean_files = glob.glob(data_dir+"*/clean/*.dat*")
        error_files = glob.glob(data_dir+"*/error/*.out*")
        self.files = clean_files + error_files
        self.clean_labels, self.error_labels = len(clean_files), len(error_files)
        self.labels = np.append(np.zeros(self.clean_labels), np.ones(self.error_labels))
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.files) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.files[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_y = self.labels[idx*self.batch_size: (idx+1)*self.batch_size]
        data = []
        for filename in batch_x:
            data.append(np.fromfile(filename, dtype=np.double).reshape(NX, NY, 1))
        return np.array(data), batch_y

    def get_true_labels(self):
        truth = [0] * (self.clean_labels/121) + [1] * (self.error_labels/121)
        return np.array(truth)


model = Sequential([
    Conv2D(46, (3,3), activation='relu', input_shape=(NX, NY, 1)),
    #Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.25),
    Flatten(),
    #Dense(256, activation='relu'),
    #Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

def compute_metrics(pred_labels, true_labels):
    clean_samples = np.sum(true_labels == 0) * 1.0
    error_samples = np.sum(true_labels == 1) * 1.0
    total = clean_samples + error_samples * 1.0
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
    data_gen = FlashDatasetGenerator(sys.argv[1], 128)
    #model.load_weights('model_keras.h5')
    #model.fit_generator(generator=data_gen, use_multiprocessing=True, workers=8, epochs=10)
    #model.save_weights('model_keras.h5')

    model.load_weights('model_keras.h5')
    #scores = model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8)
    #print scores
    scores = model.predict_generator(generator=data_gen, use_multiprocessing=True, workers=8, verbose=1)
    print scores.shape
    scores = scores.reshape((len(scores)/121, 121))
    print scores.shape
    scores = (scores >= 0.8)    # shape of (N, 121)
    pred = np.any(scores, axis=1)
    print pred.shape
    print pred
    compute_metrics(pred, data_gen.get_true_labels())

