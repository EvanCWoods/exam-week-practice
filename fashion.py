import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow_datasets as tfds

matplotlib.use('macosx')

train_ds, test_ds = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True)

train_data = []
train_labels = []

for example, label in train_ds:
    train_data.append(example.numpy())
    train_labels.append(label.numpy())

test_data = []
test_labels = []

for example, label in test_ds:
    test_data.append(example.numpy())
    test_labels.append(label.numpy())


train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

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
            print('Stopping Training')
            self.model.stop_training = True

callback = MyCallback()


def show_img(image, label):
    plt.figure(figsize=(5, 3))
    plt.title(label)
    plt.imshow(image)
    plt.show()


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
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
                    epochs=1,
                    validation_data=(validation_data, validation_labels),
                    batch_size=30,
                    callbacks=[callback])


class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def predict(model, data, labels):
    i = random.randint(0, int(test_data.shape[0]))
    data = data[i]
    label = labels[i]

    prediction = model.predict(np.reshape(data, (1, 28, 28, 1)))

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