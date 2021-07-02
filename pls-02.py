import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

matplotlib.use('macosx')

path = '../data/PLS.AX.csv'
df = pd.read_csv(path)
features = np.array(df['Close'].tolist())
targets = np.array(df['Close'].tolist())
time = np.arange(0, int(len(features)), 1)


def show_data(title, data):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.plot(data)
    plt.show()


show_data(title='Close Price Data', data=features)


split = 3000
train_features = features[:split]
train_targets = targets[:split]
train_time = time[:split]
test_features = features[split:]
test_targets = targets[split:]
test_time = time[split:]

train_features = tf.expand_dims(train_features, -1)
test_features = tf.expand_dims(test_features, -1)
print(train_features.shape)
print(test_features.shape)


def show_split(title, train_time, train_data, test_time, test_data):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.plot(train_time, train_data, label='train data')
    plt.plot(test_time, test_data, label='test data')
    plt.legend()
    plt.show()


show_split(title='train & test data', train_time=train_time, train_data=train_features,
           test_time=test_time, test_data=test_features)

LENGTH = 2
BATCH_SIZE = 32

train_generator = TimeseriesGenerator(data=train_features,
                                      targets=train_targets,
                                      length=LENGTH,
                                      batch_size=BATCH_SIZE)


model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu', input_shape=[None, 1]),
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

history = model.fit(train_generator, epochs=25)


def plot_metrics(history, title, metric):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.plot(history.history[metric], label=metric)
    plt.legend()
    plt.show()


plot_metrics(history=history, title='MsE Plot', metric='mse')


def predict_and_plot_predictions(model, data, time):

    predictions = model.predict(tf.expand_dims(data, -1))

    plt.figure(figsize=(15, 7))
    plt.title('Predictions vs Truth')
    plt.plot(time, predictions, label='predictions')
    plt.plot(time, data, label='truth')
    plt.legend()
    plt.show()


predict_and_plot_predictions(model=model, data=test_features, time=test_time)