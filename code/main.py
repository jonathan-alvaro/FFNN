import os
from math import ceil

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from layers import Dense, Sigmoid
from model import Model
from optimizer import GradientDescentOptimizer


def load_data():
    data_path = os.path.join('../', 'data/', 'weather.csv')
    data = pd.read_csv(data_path)
    for column in data.columns:
        if data[column].dtype == 'object':
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])

        elif data[column].dtype == 'bool':
            data[column] = data[column].astype(int)

        else:
            data[column] = data[column]

    X = data[data.columns[:-1]].to_records()
    X = np.array([list(data) for data in X])
    y = data[data.columns[-1]].values.reshape(-1, 1)
    return X, y


def create_model(hidden_layer, nb_nodes):
    if hidden_layer <= 0:
        raise ValueError("Model needs at least 1 hidden layer")

    elif hidden_layer > 10:
        raise ValueError("Maximum number of hidden layers is 10")

    model = Model()
    input_length = 5
    for i in range(hidden_layer):
        model.append_layer(Dense(
            node_count=nb_nodes[i], input_length=input_length)
        )
        model.append_layer(Sigmoid())
        input_length = nb_nodes[i]

    return model


def train_model(model, optimizer, X, y, batch_size, epoch):
    batch_count = len(X) / batch_size

    for i in range(epoch):
        for j in range(ceil(batch_count)):
            if j != (ceil(batch_count) - 1):
                batch_X = X[j*batch_size:(j+1)*batch_size]
                batch_y = y[j*batch_size:(j+1)*batch_size]
            else:
                batch_X = X[j * batch_size:]
                batch_y = y[j * batch_size:]

        optimizer.optimize_model(model, batch_X, batch_y)

    return model

model = create_model(2, [3, 1])
optimizer = GradientDescentOptimizer(learning_rate=0.01, momentum=0.001)
X, y = load_data()
model = train_model(model, optimizer, X, y, 5, 10)
