import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import csv
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

matplotlib.use('macosx')


train_data = []
train_time = []
test_data = []
test_time = []

with open('../data/archive-2/DailyDelhiClimateTest.csv') as train:
    reader = csv.reader(train, delimiter=',')
    next(reader)
    for row in reader:
        test_time.append(row[0])
        test_data.append(round(float(row[1]), ndigits=2))


with open('../data/archive-2/DailyDelhiClimateTrain.csv') as test:
    reader = csv.reader(test)
    next(reader)
    for row in reader:
        train_time.append(row[0])
        train_data.append(round(float(row[1]), ndigits=2))


train_data = np.array(train_data)
test_data = np.array(test_data)


print(train_data[:5])
print('train time: ', len(train_time))
print()
print(test_data[:5])
print('test time: ', len(test_time))


time_count = (int(len(train_time)) + int(len(test_time)))
total_time = np.arange(0, time_count, 1)


def plot_data(data, time, title):
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.plot(time, data)
    plt.show()


plot_data(data=train_data, time=train_time, title='train data')
plot_data(data=test_data, time=test_time, title='test data')


train_data = tf.expand_dims(train_data, -1)
test_data = tf.expand_dims(test_data, -1)

train_features = train_data
train_targets = train_data
test_features = test_data
test_targets = test_data


BATCH_SIZE = 32
INPUT_LENGTH = 11


train_generator = TimeseriesGenerator(data=train_features,
                                      targets=train_targets,
                                      sampling_rate=1,
                                      stride=1,
                                      batch_size=BATCH_SIZE,
                                      length=INPUT_LENGTH)


tf.random.set_seed(42)
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
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])

history = model.fit(train_generator, epochs=50)


predictions = []

def predict(model, data, labels, time):

    prediction = model.predict(data)

    label = labels[11:]

    plt.figure(figsize=(20, 5))
    plt.plot(time, prediction, label='pred')
    plt.plot(time, label, label='true')
    plt.legend()
    plt.show()

    predictions.append(prediction)


predict(model=model, data=TimeseriesGenerator(data=test_data,
                                              targets=test_data,
                                              sampling_rate=1,
                                              stride=1,
                                              batch_size=BATCH_SIZE,
                                              length=INPUT_LENGTH),
        labels=test_data, time=test_time[11:])



def plot_history(history, metric):
    plt.figure(figsize=(15, 5))
    plt.plot(history.history[metric])
    plt.show()

plot_history(history=history, metric='mse')


model.save('climate-lstm')