import numpy as np

from layers import Sigmoid

class GradientDescentOptimizer(object):
    def __init__(self, learning_rate=0.01, momentum=0):
        self._lr = learning_rate
        self._momentum = momentum

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        self._lr = new_learning_rate

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, new_momentum):
        self._momentum = new_momentum

    def optimize_model(self, model, input_matrix, labels):
        output = model(input_matrix)
        error = labels - output
        layers = model.layers

        output_layer = True

        for layer in reversed(layers):
            if type(layer) == Sigmoid:
                continue

            if output_layer:
                delta = error * output * (1 - output)
                update_delta = self.learning_rate * delta
                update = np.matmul(
                    update_delta.transpose(),
                    layer.prev_input
                ).transpose()
                output_layer = False

            else:
                delta = np.matmul(
                    prev_delta, prev_layer.weights.transpose()
                )
                update_delta = delta
                update = np.matmul(
                    update_delta.transpose(),
                    layer.prev_input
                ).transpose()

            # print(update)


            if layer.prev_weights is not None:
                layer.weights = layer.weights = layer.weights + update\
                    +layer.prev_weights * self.momentum
            else:
                layer.weights = layer.weights + update
            prev_delta = update_delta
            prev_layer = layer
