import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
matplotlib.use('macosx')


path = 'PLS.AX.CSV'
dataframe = pd.read_csv(path)

features = dataframe[['Open', 'High', 'Low', 'Close']].to_numpy().tolist()
targets = dataframe['Close'].tolist()

split_time = 3200
train_features = features[:split_time]
train_targets = targets[:split_time]
test_features = features[split_time:]
test_targets = targets[split_time:]


# Shows all four of the dimensions of the input data
def show_data(train_data, test_data, train_time, test_time):
    plt.figure(figsize=(20, 10))
    plt.title('Data')
    plt.plot(train_time, train_data, label='training')
    plt.plot(test_time, test_data, label='testing')
    plt.legend()
    plt.show()


# show_data(train_features, test_features, train_time, test_time)


LENGTH = 3
BATCH_SIZE = 3


inputs = TimeseriesGenerator(data=train_features,
                             targets=train_targets,
                             length=LENGTH,
                             batch_size=BATCH_SIZE)


test_input_series = TimeseriesGenerator(data=test_features,
                                        targets=test_targets,
                                        length=LENGTH,
                                        batch_size=BATCH_SIZE)



def build_model():
    tf.random.set_seed(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='causal', input_shape=[None, 4]),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])

    return model


model = build_model()
history = model.fit(inputs, epochs=25)

# model.save('/SAVED-MODEL')


preds = []


def predict(model, data, pred_data):
    predictions = model.predict(pred_data)

    plt.plot(predictions, label='predicted')
    plt.plot(data, label='real')
    plt.legend()
    plt.show()


predict(model=model, data=test_targets, pred_data=test_input_series)

print(preds)
