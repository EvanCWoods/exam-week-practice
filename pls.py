import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import csv
import random
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

matplotlib.use('macosx')

path = '../data/PLS.AX.csv'

df = pd.read_csv(path)

features = df[['Open', 'High', 'Low', 'Close']].to_numpy().tolist()
targets = df['Close'].tolist()


split_point = 3000
train_features = features[:split_point]
train_targets = targets[:split_point]
test_features = features[split_point:]
test_targets = targets[split_point:]

LENGTH = 1
BATCH_SIZE = 32


ts_gen = TimeseriesGenerator(data=train_features,
                             targets=train_targets,
                             length=LENGTH,
                             batch_size=BATCH_SIZE)


test_input_series = TimeseriesGenerator(data=test_features,
                                        targets=test_targets,
                                        length=LENGTH,
                                        batch_size=BATCH_SIZE)



model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu', input_shape=[None, 4]),
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
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])

history = model.fit(ts_gen, epochs=10)



def predict(model, data, pred_data):
    predictions = model.predict(pred_data)


    plt.plot(predictions, label='predicted')
    plt.plot(data, label='real')
    plt.legend()
    plt.show()

predict(model=model, data=test_features, pred_data=test_input_series,)