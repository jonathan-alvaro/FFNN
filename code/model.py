from math import log
from typing import List

from layers import Dense, Sigmoid


class Model(object):
    def __init__(self, learning_rate=0.9, momentum=0):
        self._layers = []
        self._learning_rate = learning_rate
        self._momentum = momentum

    def _verify_input(self, input_matrix):
        if len(input_matrix.shape) != 2:
            raise ValueError(
                (
                    "Expected 2D matrix, "
                    "got array of shape {} instead"
                ).format(input_matrix.shape)
            )

        if input_matrix.shape[-1] != self.input_length:
            raise ValueError(
                (
                    "Expected matrix of shape (n, {}), "
                    "got one of shape {} instead"
                ).format(self.input_length, input_matrix.shape[-1])
            )

    def __call__(self, input_matrix):
        self._verify_input(input_matrix)

        output = input_matrix
        for layer in self.layers:
            output = layer(output)

        return output

    def _verify_new_layer(self, new_layer, prev_layer):

        if type(new_layer) == Sigmoid:
            return

        if new_layer.input_length != prev_layer.node_count:
            raise ValueError(
                (
                    "Invalid input length for new layer, "
                    "should be {} got {} instead"
                ).format(prev_layer.node_count, new_layer.input_length)
            )

    def append_layer(self, new_layer):
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            if type(prev_layer) == Sigmoid:
                prev_layer = self.layers[-2]
            self._verify_new_layer(new_layer, prev_layer)
        self._layers.append(new_layer)

    @property
    def depth(self):
        return len(self._layers)

    @property
    def layers(self) -> List[Dense]:
        return self._layers

    @property
    def weights(self):
        return [l.weights for l in self.layers]

    @property
    def input_length(self):
        if len(self.layers) == 0:
            return 0
        return self.layers[0].input_length

    def cost_fn(self, x, y):
        output = self(x)
        cost = 0
        for label, pred in zip(y, output):
            try:
                if label == 1:
                    cost += log(pred)
                else:
                    cost += log(1 - pred)
            except ValueError:
                cost += 100
        return cost / len(y)