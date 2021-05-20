from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from copyfile import *
from matplotlib import pyplot as plt

# copy dataset to temporary location
# copy_images()

# data augmentation
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

# base model for transfer learning
# pretrained vgg16 model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=img_shape
)

# main sequential model
vgg_model = Sequential()
vgg_model.add(base_model)
vgg_model.add(Dropout(0.3))
vgg_model.add(Flatten())
vgg_model.add(Dropout(0.5))
vgg_model.add(Dense(1, activation='sigmoid'))

# don't train the model
vgg_model.layers[0].trainable = False

vgg_model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

# generate model summary
vgg_model.summary()

# training
epochs = 120
vgg_history = vgg_model.fit(
    training_set,
    epochs=epochs
)

# save model if need
# model.save('/path/to/model/folder')

# result plotting
plt.plot(vgg_history.history['loss'], 'r', label='train')
plt.plot(vgg_history.history['accuracy'], 'g', label='validation')
plt.legend(['loss', 'accuracy'])
plt.xlabel('epochs')
plt.ylabel('measure')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/model/vgg_run1.png', bbox_inches='tight')
