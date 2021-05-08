from os import listdir, makedirs

import cv2 as cv
import tensorflow as tk
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from keras_preprocessing.image import ImageDataGenerator


def data_augmentation(file_dir, num_samples, save_to):
    data_gen = ImageDataGenerator(rotation_range=10,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  brightness_range=(0.3, 1.0),
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

    for filename in listdir(file_dir):
        img = cv.imread(file_dir + '/' + filename)
        img = img.reshape((1,) + img.shape)
        save_prefix = 'aug_' + filename[:-4]
        i = 0
        for batch in data_gen.flow(x=img,
                                   batch_size=1,
                                   save_to_dir=save_to,
                                   save_prefix=save_prefix,
                                   save_format='jpg'):
            i += 1
            if i > num_samples:
                break

data_augmentation('dataset/yes/', 6, '/tmp/augemented_data/')
