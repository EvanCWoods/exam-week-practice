import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

url = 'https://homl.info/shakespeare'

filepath = tf.keras.utils.get_file('shakespeare.txt', url)

with open(filepath) as f:
    text = f.read()

corpus = text.lower()

tokenizer =  Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index


line_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(0, len(token_list)):
        n_gram_sequences = token_list[:i+1]
        line_sequences.append(n_gram_sequences)

max_sequence_length = max([len(x) for x in line_sequences])
input_sequences = pad_sequences(line_sequences,
                                maxlen=max_sequence_length + 1,
                                padding='pre')


x = input_sequences[:, :-1]
y = input_sequences[:, -1]
labels = tf.keras.utils.to_categorical(y, num_classes=10000)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_sequence_length),
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
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(5000, activation='relu'),
    tf.keras.layers.Dense(10000, activation='softmax'),
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x, labels,
                    batch_size=32,
                    steps_per_epoch=1000,
                    epochs=1)


seed_text = 'for you'
next_words = 25

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print()
print()
print(seed_text)