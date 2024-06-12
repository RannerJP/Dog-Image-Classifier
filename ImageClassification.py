#%%
import numpy as np
from matplotlib import pyplot as plot
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import keras.metrics
import os
clusters = [16,32,64]
convulationLayers = [1,2]
convulationFilters = [16, 32, 64]
epochs = [5,10,15,20]
'''
Classification of breeds is according to the folder name:

0 - Borzoi
1 - Chihuahua
2 - Dingo
3 - German Shepard
4 - Golden Retriever
5 - Mexican Hairless
6 - Pug
7 - Shih-Tzu
8 - Siberian Husky
9 - (Standard) Poodle

^^This code was to see what each item was labeled as in order to 
'''
best_accuracy = 0
best_model = Sequential()
for cluster_size in clusters:
    for layers in convulationLayers:
        for filters in convulationFilters:
            for steps in epochs:
                data = tf.keras.utils.image_dataset_from_directory("images", batch_size=cluster_size)
                data = data.map(lambda x,y: (x/255, tf.one_hot(y, 10)))
                training_size = round(len(data) * .7)
                validation_size = round(len(data) * .2)
                test_size = round(len(data) * .1)
                training = data.take(training_size)
                validation = data.skip(training_size).take(validation_size)
                test = data.skip(training_size + validation_size).take(test_size)
                model = Sequential()
                model.add(Conv2D(filters, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
                model.add(MaxPooling2D())
                for i in range(layers-1):
                    model.add(Conv2D(filters, (3,3), 1, activation='relu'))
                    model.add(MaxPooling2D())
                model.add(Flatten())

                model.add(Dense(256, activation='relu'))
                model.add(Dense(10, activation='softmax'))
                
                model.compile(loss='categorical_crossentropy', metrics=('accuracy'), optimizer='adam')
                model.fit(training, epochs=steps, validation_data=(validation), batch_size=cluster_size)
                accuracy = keras.metrics.CategoricalAccuracy()
                for batch in test.as_numpy_iterator():
                    X, y = batch
                    prediction = model.predict(X)
                    accuracy.update_state(y, prediction)
                if accuracy.result().numpy() > best_accuracy:
                    best_accuracy = accuracy.result().numpy()
                    best_model = model
                    print(f"best model so far with accuracy: {accuracy.result().numpy()}, with clusters of size: {cluster_size}, convolution layers: {layers}, # of Filters {filters}, # of epochs: {steps}")
best_model.save(os.path.join('models', 'DogClassification.h5'))
# %%
