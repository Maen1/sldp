import os
# Imports for Deep Learning
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
# Ensure consistency across runs
from numpy.random import seed
import random
seed(2)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow import set_random_seed
set_random_seed(2)

import cv2
from glob import glob


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tensorflow 
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True


# Set global variables
data_dir = "../asl_alphabet/asl_alphabet_train"
test_dir = '../asl_alphabet/asl_alphabet_test'

target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
#test_generator = data_augmentor.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="testing")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

# CNN model

my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(n_classes, activation='softmax'))

parallel_model = multi_gpu_model(my_model, gpus=8)
parallel_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

parallel_model.fit_generator(train_generator,steps_per_epoch=train_generator.__len__(), epochs=5, validation_data=val_generator, validation_steps=batch_size)

parallel_model.save('test.h5')

# score = cnn_model.evaluate(test_generator, verbose=1)
# print('test loss : {:.4f}'.format(score[0]))
# print('test acc : {:.4f}'.format(score[1]))

cnn_model.summary()

print("done")
