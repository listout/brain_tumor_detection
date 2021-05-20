import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from copyfile import *

# copy_images()

model = tf.keras.models.load_model('model/my_model.h5')

print(model.summary())

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(.3, 1.),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# vgg16 specific settings
img_size = (224, 224)
img_shape = img_size + (3,)
random_seed = 123

# training set
training_set = train_datagen.flow_from_directory(
    training_dataset_directory,
    color_mode='rgb',
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    seed=random_seed,
    subset='training'
)

# testing set
testing_set = train_datagen.flow_from_directory(
    testing_dataset_directory,
    target_size=img_size,
    batch_size=16,
    class_mode='binary',
    seed=random_seed,
    subset='validation'
)

loss, acc = model.evaluate(training_set)
