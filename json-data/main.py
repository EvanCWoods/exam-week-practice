import json
import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
import pandas as pd
import os


with open('data/iris.json') as f:
    dataset = json.load(f)

dataframe = pd.read_json('data/iris.json')

dataframe['species'] = pd.Categorical(dataframe['species'])
dataframe['species'] = dataframe.species.cat.codes

features = dataframe[['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']]
targets = dataframe['species']

features = np.array(features).astype(float)
targets = np.array(targets).astype(float)

print(features.shape)
print(targets.shape)

inputs = tf.data.Dataset.from_tensor_slices((features, targets)).shuffle(2).batch(1)

print(inputs)

def build_model():

    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=[4]),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = build_model()
history = model.fit(inputs, epochs=50)

