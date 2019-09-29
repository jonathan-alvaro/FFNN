import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from optimizer import GradientDescentOptimizer
from layers import Dense, Sigmoid
from model import Model


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

X, y = load_data()
model = create_model(2, [5, 1])

for i in range(10000):
    if i % 1000 == 0:
        print("Accuracy:", ((model(X) > 0.5).astype(int) == y).sum() / len(X))
    optimizer = GradientDescentOptimizer(learning_rate=0.003)
    optimizer.optimize_model(model, X, y)

print("Accuracy:", ((model(X) > 0.5).astype(int) == y).sum() / len(X))