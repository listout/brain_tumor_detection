from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from matplotlib import pyplot as plt
from copyfile import *

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

# vgg specific settings
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

# base vgg model
vgg = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=img_shape
)

# main sequential model
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# don't train the resent model
model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

# generate model summary
model.summary()

# training the trainable paramerters
epochs = 30
vgg_hist = model.fit(
    training_set,
    epochs=epochs,
    validation_data=testing_set
)

# save model if need
# model.save('/path/to/model/folder')

# metrics
# acc = vgg_hist.history['accuracy']
# val_acc = vgg_hist.history['val_accuracy']
# loss = vgg_hist.history['loss']
# val_loss = vgg_hist.history['val_loss']

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
# plt.savefig('/path/to/save/.png', bbox_inches='tight')
