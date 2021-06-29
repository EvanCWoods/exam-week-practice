import tensorflow as tf
import numpy as np
import random
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train_ds, test_ds = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

train_data = []
train_labels = []

for example, label in train_ds:
    train_data.append(example.numpy().decode('utf-8'))
    train_labels.append(label.numpy())


test_data = []
test_labels = []

for example, label in test_ds:
    test_data.append(example.numpy().decode('utf-8'))
    test_labels.append(label.numpy())

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
word_index = tokenizer.word_index

input_sequences = pad_sequences(train_sequences,
                                maxlen=256,
                                padding='post')


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            print('Stopping training')
            self.model.stop_training = True


callback = MyCallback()


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 32, input_length=256),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    ),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(input_sequences, train_labels,
                    epochs=1,
                    callbacks=[callback])


def predict(model, data, labels):
    i = random.randint(0, int(test_data.shape[0]))
    data = data[i]
    label = labels[i]

    test_sequences = tokenizer.texts_to_sequences(data)
    test_sequences = pad_sequences(test_sequences,
                                   maxlen=256,
                                   padding='post')

    prediction = model.predict(test_sequences)

    if prediction.argmax() > 0.5:
        prediction = 1
    else:
        prediction = 0

    print()
    print(data)
    print('Predicted: ', prediction)
    print('Label: ', label)

    if prediction - label == 0:
        print('Correct')
    else:
        print('Incorrect')


def predict_cycle():
    for _ in range(10):
        predict(model=model, data=test_data, labels=test_labels)

predict_cycle()