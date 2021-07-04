import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

path = 'https://www.gutenberg.org/files/65753/65753-0.txt'
file = tf.keras.utils.get_file('65753-0.txt', path)
text = open(file, 'rb',).read().decode(encoding='utf-8')


corpus = text.lower().split('.')

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1


inputs = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequences = token_list[:i+1]
        inputs.append(n_gram_sequences)

max_sequence_length = max([len(x) for x in inputs])
inputs = pad_sequences(inputs, maxlen=max_sequence_length, padding='pre')

x = inputs[:, :-1]
y = inputs[:, -1]
labels = tf.keras.utils.to_categorical(y, num_classes=total_words)

EMBEDDING_DIMS = 16


def build_model():
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, EMBEDDING_DIMS, input_length=max_sequence_length),
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
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(total_words, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = build_model()
history = model.fit(x, labels, epochs=25)


def predict(model, seed, n_words):
    seed = seed
    n_words = n_words

    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed += " " + output_word

    print()
    print(seed)


predict(model=model, seed='This is an', n_words=30)
