import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
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

batch_size = 64
epochs=150
num_classes = 10

weight_decay=.0001
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train[:,:,:,0] -=np.mean(x_train[:,:,:,0])
x_train[:,:,:,1] -=np.mean(x_train[:,:,:,1])
x_train[:,:,:,2] -=np.mean(x_train[:,:,:,2])
x_test[:,:,:,0] -=np.mean(x_test[:,:,:,0])
x_test[:,:,:,1] -=np.mean(x_test[:,:,:,1])
x_test[:,:,:,2] -=np.mean(x_test[:,:,:,2])
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def scheduler(epoch):
	if epoch<=50:
		return 0.05
	if epoch<=90:
		return 0.01
	if epoch<=120:
		return 0.002
	return 0.0004

model = Sequential()
model.add(Conv2D(64, (5, 5), padding='same',kernel_regularizer=regularizers.l2(weight_decay),input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
model.add(Dropout(0.3))


model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(1024, (1,1), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(1024, (1, 1), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(10, (1, 1), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())


model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))
"""
model.add(Conv2D(2048, (4, 4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(1024, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(10, (1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))"""




"""
model.add(Flatten())
model.add(Dense(units = 1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = num_classes,activation='softmax'))"""

sgd = optimizers.SGD(lr=.01,momentum=0.9,nesterov=True)
# Compiling the CNN
model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
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
"""
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']


xc=range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('Training loss vs Validation loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('new4/loss.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['Training Accuracy','Validation Accuracy'],loc=4)
plt.style.use(['classic'])
plt.savefig('new4/acc.png')

plt.figure(3, figsize=(7,5))
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc')
plt.grid(True)
plt.savefig('new4/val_acc.png')

#print plt.style.available # use bmh, classic,ggplot for big pictures

"""
from keras.utils import plot_model
plot_model(model,to_file='/output/new4/model.png')
"""
git push -u origin master
git remote add origin https://github.com/mohit8935/Final-Year-Project.git

"""
