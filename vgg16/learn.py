

import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.feature import hog
import datetime as dt

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Flatten,Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras import regularizers
from keras.applications import VGG16, VGG19
import cv2


num_classes =10
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def resize_data(data, size_list):
    data_upscaled = np.zeros(tuple(size_list))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=tuple(size_list[1:3]), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

x_train = resize_data(x_train, [x_train.shape[0], 48, 48, 3])/255
x_test = resize_data(x_test, [x_test.shape[0], 48, 48, 3])/255
print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

vgg16_model = VGG16(weights='imagenet', include_top=False,input_shape=(48,48,3))
x = vgg16_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)
sgd = optimizers.SGD(lr=.01,momentum=0.9,nesterov=True)
for layer in vgg16_model.layers:
layer.trainable = False

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
t_datagen = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125)
t_datagen.fit(x_train)

tensorboard = TensorBoard(log_dir='/output/new4', histogram_freq=0, write_graph=True, write_images=True)


hist = model.fit_generator(t_datagen.flow(x_train, y_train,
                                          batch_size=batch_size),steps_per_epoch = x_train.shape[0]/batch_size,
                           epochs=epochs,
                           validation_data=(x_test, y_test),validation_steps = x_test.shape[0]/batch_size,
                           callbacks=[tensorboard,LearningRateScheduler(scheduler)]
                           )

model.save_weights('/output/model.h5')
model.save('/output/model.hdf5')

scores = model.evaluate(x_test,y_test)
print(scores[0])
print(scores[1])





