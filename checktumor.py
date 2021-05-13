import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.3, 1.0),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

training_directory = '/tmp/dataset/training'
testing_directory = '/tmp/dataset/testing'

training_set = train_datagen.flow_from_directory(
    training_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# test_datagen = ImageDataGenerator(
# rescale=1. / 255
# )

test_set = train_datagen.flow_from_directory(
    testing_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

cnn = tf.keras.models.Sequential()
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=[150, 150, 3]
    )
)
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2
    )
)
cnn.add(
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
    )
)
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2
    )
)
cnn.add(
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu',
    )
)
cnn.add(
    tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2
    )
)
cnn.add(
    tf.keras.layers.Dropout(
        0.2
    )
)
cnn.add(
    tf.keras.layers.Flatten()
)
cnn.add(
    tf.keras.layers.Dense(
        units=512,
        activation='relu'
    )
)
cnn.add(
    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid'
    )
)

cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# cnn.summary()

train_steps_per_epoch = training_set.n // training_set.batch_size
test_steps_per_epoch = test_set.n // test_set.batch_size

epochs = 100

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
cnn.save('models')

history = cnn.fit(
    training_set,
    validation_data=test_set,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    callbacks=[cp_callback]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

figs, ax = plt.subplots(2)
ax[0].plot(epochs, acc, 'r', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'b', label='Validation Accuracy')
ax[0].set_title('Training and Validation Accuracy')
ax[0].legend(loc='lower right')

ax[1].plot(epochs, loss, 'r', label='Training Loss')
ax[1].plot(epochs, val_loss, 'b', label='Validation Loss')
ax[1].set_title('Training and Validation Loss')
ax[1].legend(loc='upper right')


figs.tight_layout()
figs.savefig('test_run4.png', bbox_inches='tight')
