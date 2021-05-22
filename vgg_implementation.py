from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from copyfile import *
from matplotlib import pyplot as plt

# data augmentation
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
    epochs=epochs,
    validation_data=testing_set
)

# save model if need
# model.save('/path/to/model/folder')

# metrics
# acc = vgg_history.history['accuracy']
# val_acc = vgg_history.history['val_accuracy']
# loss = vgg_history.history['loss']
# val_loss = vgg_history.history['val_loss']

# result plotting
# figs, ax = plt.subplots(2, sharex=True)
# ax[0].plot(acc, label='Training Accuracy')
# ax[0].plot(val_acc, label='Validation Accuracy')
# ax[0].legend(loc='lower right')
# ax[0].set_ylabel('Accuracy')
# ax[0].set_title('Training and Validation Accuracy')

# ax[1].plot(loss, label='Training Loss')
# ax[1].plot(val_loss, label='Validation Loss')
# ax[1].legend(loc='upper right')
# ax[1].set_ylabel('Cross Entropy')
# ax[1].set_title('Training and Validation Loss')
# ax[1].set_xlabel('epoch')

# plt.tight_layout()
# plt.savefig('/content/drive/MyDrive/model/vgg_run1.png', bbox_inches='tight')
