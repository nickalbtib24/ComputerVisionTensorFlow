import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


K.clear_session()


data_entrenamiento = '/tmp/data/entrenamiento'
data_validacion = '/tmp/data/validacion'


epocas=4
longitud, altura = 300, 300
batch_size = 45
pasos = 700
validation_steps = 600
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 16
lr = 0.0004



entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')



validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

sample_training_images, _ = next(entrenamiento_generador)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(np.squeeze(img))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])


cnn = Sequential()
cnn.add(Convolution2D(32, (3,3), padding ="same", input_shape=(300, 300, 3), activation='relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Convolution2D(64, (3,3), padding ="same", input_shape=(300, 300, 3), activation='relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Convolution2D(128, (3,3), padding ="same", input_shape=(300, 300, 3), activation='relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Convolution2D(128, (3,3), padding ="same", input_shape=(300, 300, 3), activation='relu'))
cnn.add(MaxPooling2D((2,2)))
cnn.add(Flatten())
cnn.add(Dropout(0.5))
cnn.add(Dense(512, activation='relu'))
cnn.add(Dense(16, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['accuracy'])




cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
   validation_steps=validation_steps)

target_dir = './modelo/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')



