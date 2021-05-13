import os

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def augement_data(file_dir, n_general_samples, save_to_dir):

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for filename in os.listdir(file_dir):
        image = cv2.imread(file_dir + '/' + filename)
        image = image.reshape((1,) + image.shape)
        save_prefix = 'aug_' + filename[:-4]

        i = 0

        for batch in datagen.flow(x=image, batch_size=1,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_to_dir, save_format='jpg'):
            i += 1
            if i > n_general_samples:
                break


augement_data(file_dir='dataset/yes', n_general_samples=6,
              save_to_dir='/tmp/images/training/yes/')
augement_data(file_dir='dataset/no', n_general_samples=9,
              save_to_dir='/tmp/images/training/no/')
