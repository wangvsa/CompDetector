from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D
import keras
import numpy as np
import sys, glob

NX, NY = 60, 60

class FlashDatasetGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size):
        clean_files = list(glob.iglob(data_dir+"*/clean/*.dat"))
        error_files = list(glob.iglob(data_dir+"*/error/*.out"))
        self.files = clean_files + error_files
        self.labels = np.append(np.zeros(len(clean_files)), np.ones(len(error_files)))
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

if __name__ == "__main__":
    data_gen = FlashDatasetGenerator(sys.argv[1], 64)
    model.load_weights('model_keras.h5')
    model.fit_generator(generator=data_gen, use_multiprocessing=True, workers=8, epochs=20)
    model.save_weights('model_keras.h5')

    #model.load_weights('my_model_weights.h5')
    #scores = model.evaluate_generator(generator=data_gen, use_multiprocessing=True, workers=8)
    #print scores

