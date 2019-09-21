from typing import List

from code.layers import Dense
import numpy as np


class Model(object):
    def __init__(self):
        self._layers = []

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
        prev_layer = self.layers[-1]
        if new_layer.input_length != prev_layer.node_count:
            raise ValueError(
                (
                    "Invalid input length for new layer, "
                    "should be {} got {} instead"
                ).format(prev_layer.node_count, new_layer.input_length)
            )

    def append_layer(self, new_layer: Dense):
        if len(self.layers) > 0:
            self._verify_new_layer(new_layer, self.layers[-1])
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

    def _calculate_loss(self, input_matrix, label_array):
        model_output = self(input_matrix)
        squared_error = np.square(model_output - label_array)
        return np.mean(squared_error)
