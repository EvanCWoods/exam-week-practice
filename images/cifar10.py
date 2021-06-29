import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import random
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use('macosx')

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()

train_data = train_data / 255
test_data = test_data / 255

validation_split = int(test_data.shape[0] * 0.2)
test_data = test_data[validation_split:]
test_labels = test_labels[validation_split:]
validation_data = test_data[:validation_split]
validation_labels = test_labels[:validation_split]


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print('Stopping Training')
            self.model.stop_training = True

callback = MyCallback()

def show_img(image, label):
    plt.figure(figsize=(5, 3))
    plt.title(label)
    plt.imshow(image)
    plt.show()


train_datagen = ImageDataGenerator(rotation_range=20,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   shear_range=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(train_data, train_labels, batch_size=32),
                    epochs=10,
                    validation_data=(validation_data, validation_labels),
                    callbacks=[callback])


class_names = (['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])


def predict(model, data, labels):
    i = random.randint(0, int(test_data.shape[0]))
    data = data[i]
    label = int(labels[i])

    prediction = model.predict(np.reshape(data, (1, 32, 32, 3)))

    print()
    print('Prediction: ', class_names[prediction.argmax()])
    print('Label: ', class_names[label])

    if prediction.argmax() - label == 0:
        print('Correct')
    else:
        print('Incorrect')

    show_img(image=data, label=class_names[label])


def predict_cycle():
    for _ in range(10):
        predict(model=model, data=test_data, labels=test_labels)


predict_cycle()