import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import data_augmentation as da

"""augmentation"""

# DATA_DIR = '/tmp/images'
# da.makedirs(DATA_DIR + '/yes')
# da.makedirs(DATA_DIR + '/no')

# da.data_augmentation('dataset/yes', 6, DATA_DIR + '/yes')
# da.data_augmentation('dataset/no', 9, DATA_DIR + '/no')

datagen = ImageDataGenerator(validation_split=0.2,
                             rescale=1. / 255, fill_mode='nearest')

TRAIN_DIR = '/tmp/images'

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    subset='validation'
)

model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation='relu',
        input_shape=(150, 150, 3)
    )
)
model.add(
    tf.keras.layers.MaxPool2D(2, 2)
)

model.add(
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    )
)
model.add(
    tf.keras.layers.MaxPool2D(2, 2)
)

model.add(
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)
model.add(
    tf.keras.layers.MaxPool2D(2, 2)
)

model.add(
    tf.keras.layers.Flatten()
)
model.add(
    tf.keras.layers.Dense(
        512,
        activation='relu')
)
model.add(
    tf.keras.layers.Dense(
        1, activation='sigmoid'
    )
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# print(model.summary())

history = model.fit(
    train_generator,
    epochs=20,
    verbose=1,
    validation_data=val_generator
)

# print(history.history.keys())
accuracy = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

epochs = range(len(accuracy))


figs, ax = plt.subplots(1)
ax[0].plot(epochs, accuracy, 'r', "Training Accuracy")
ax[0].plot(epochs, val_accuracy, 'b', "Validation Accuracy")
ax[0].set_title('Training and Validation Accuracy')

# ax[1].plot(epochs, loss, 'r', 'Training Loss')
# ax[1].plot(epochs, val_loss, 'b', 'Validation Loss')
# ax[1].set_title('Training and Validation Loss')

# figs.tight_layout()
figs.savefig('loss_accuracy.png', bbox_inches='tight')
