import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use('macosx')


def show_img(image, title):
    plt.figure(figsize=(5, 3))
    plt.title(title)
    plt.imshow(image)
    plt.show()


(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

train_data = tf.expand_dims(train_data, -1)
test_data = tf.expand_dims(test_data, -1)


train_data = train_data / 255
test_data = test_data / 255


validation_split = (int(test_data.shape[0] * 0.2))
test_data = test_data[validation_split:]
test_labels = test_labels[validation_split:]
validation_data = test_data[:validation_split]
validation_labels = test_labels[:validation_split]


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print()
            print('Stopping training')
            self.model.stop_training = True


callback = MyCallback()


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=10,
                    validation_data=(validation_data, validation_labels),
                    callbacks=[callback])


def predict(model, data, labels):
    i = random.randint(0, int(test_data.shape[0]))

    data = data[i]
    label = labels[i]

    prediction = model.predict(np.reshape(data, (1, 28, 28, 1)))

    print()
    print('Prediction: ', prediction.argmax())
    print('Label: ', label)

    if prediction.argmax() - label == 0:
        print('Correct')
    else:
        print('Incorrect')

    show_img(data, label)


def predict_cycle():
    for _ in range(10):
        predict(model=model, data=test_data, labels=test_labels)

predict_cycle()