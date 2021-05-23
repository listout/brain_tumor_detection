import sys
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tempfile
import os

# check if images are copied to temp directory
temp_dataset_dir = tempfile.gettempdir() + '/dataset'
if not os.path.isdir(temp_dataset_dir):
    print('Running test before the actual code')
    print('Run the actual code, it will copy the dataset')
    sys.exit()

# training and testing temporary directory
training_dataset_directory = temp_dataset_dir + '/training/'
testing_dataset_directory = temp_dataset_dir + '/testing/'
print('Training Dataset directory', training_dataset_directory)
print('Testing Dataset directory', testing_dataset_directory)

model = tf.keras.models.load_model('model/vgg_imp.h5')

print(model.summary())

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# vgg16 specific settings
img_size = (150, 150)
img_shape = img_size + (3,)

# training set
training_set = train_datagen.flow_from_directory(
    training_dataset_directory,
    target_size=img_size,
    class_mode='binary',
    subset='training'
)

# testing set
testing_set = train_datagen.flow_from_directory(
    testing_dataset_directory,
    target_size=img_size,
    class_mode='binary',
    subset='validation'
)

loss, acc = model.evaluate(training_set)
