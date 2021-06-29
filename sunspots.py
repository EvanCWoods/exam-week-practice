import tensorflow as tf
import numpy as np
import csv
import random
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('macosx')


path_to_data = '../data/Sunspots.csv'

sunspots = []
time = []

with open(path_to_data) as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time.append(int(row[0]))

series = np.array(sunspots)
time = np.array(time)


def show_raw_data(data, time_step, title):
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.plot(time_step, data)
    plt.show()


show_raw_data(data=sunspots, time_step=time, title='sunspots')


test_split = int(len(series) * 0.7)
train_data = series[:test_split]
train_time = time[:test_split]
test_data = series[test_split:]
test_time = time[test_split:]
val_split = int(len(series) * 0.9)
val_data = series[val_split:]
val_time = time[val_split:]


def show_split(train, test, val, train_time, test_time, val_time):
    plt.figure(figsize=(20, 5))
    plt.plot(train_time, train)
    plt.plot(test_time, test)
    plt.plot(val_time, val)
    plt.title('Split Data')
    plt.show()


show_split(train=train_data, test=test_data, val=val_data, train_time=train_time, test_time=test_time, val_time=val_time)


train_features = tf.expand_dims(train_data, -1)
train_targets = train_features
test_features = tf.expand_dims(test_data, -1)
test_targets = test_features
val_features = tf.expand_dims(val_data, -1)
val_targets = val_features

length = 11
batch_size = 32

train_input_series = TimeseriesGenerator(data=train_features,
                                         targets=train_targets,
                                         length=length,
                                         batch_size=batch_size)

test_input_series = TimeseriesGenerator(data=test_features,
                                         targets=test_targets,
                                         length=length,
                                         batch_size=batch_size)

val_input_series = TimeseriesGenerator(data=val_features,
                                         targets=val_targets,
                                         length=length,
                                         batch_size=batch_size)


model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 5, padding='causal', activation='relu', input_shape=[None, 1]),
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
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])

history = model.fit(train_input_series,
                    epochs=25,
                    validation_data=val_input_series)


def predict(model, data, pred_data, labels):
    predictions = model.predict(pred_data)

    pred_labels = labels[11:]

    plt.plot(pred_labels, predictions, label='predicted')
    plt.plot(labels, data, label='real')
    plt.legend()
    plt.show()

predict(model=model, data=test_data, pred_data=test_input_series,labels=test_time)