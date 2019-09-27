from abc import abstractmethod

import numpy as np


class Layer(object):
    def __init__(self, input_length: int = 1):
        if input_length <= 0:
            raise ValueError(
                "Invalid input array length, expected at least 1"
            )
        self._input_length = input_length

    @abstractmethod
    def __call__(self, input_matrix):
        pass

    @abstractmethod
    def _verify_input_shape(self, input_matrix):
        pass

    @property
    def input_length(self):
        return self._input_length

    @property
    def output_shape(self):
        pass


class Dense(Layer):
    def __init__(self, node_count: int = 1, input_length: int = 1):
        super().__init__(input_length=input_length)

        if node_count <= 0:
            raise ValueError(
                "Invalid value for number of nodes, expected at least 1"
            )
        self._node_count = node_count
        self._weights = np.random.random_sample(
            (self.input_length, self.node_count)
        )

    def _verify_input_shape(self, input_matrix):
        matrix_shape = input_matrix.shape
        if len(matrix_shape) != 2:
            raise ValueError(
                "Dense layers can only process 2D matrices"
            )

        if matrix_shape[-1] != self.input_length:
            raise ValueError(
                "Expected {} features, got {} instead".format(
                    self.input_length, matrix_shape[-1]
                )
            )

    def __call__(self, input_matrix):
        self._verify_input_shape(input_matrix)

        return np.matmul(input_matrix, self.weights)

    @property
    def node_count(self):
        return self._node_count

    @property
    def output_shape(self):
        return 1, self.node_count

    @property
    def weights(self):
        return self._weights


class Sigmoid(Layer):
    def __init__(self, input_length: int = 1):
        super().__init__(input_length=input_length)

    def _verify_input_shape(self, input_matrix):
        if input_matrix.shape[-1] != self.input_length:
            raise ValueError(
                (
                    "Expected arrays with {} features, "
                    "got one of shape {} instead"
                ).format(self.input_length, input_matrix.shape)
            )

    def __call__(self, input_matrix):
        return 1 / (1 + np.exp(-input_matrix))

    def output_shape(self):
        return 1, self.input_length

    def derived_fn(self, input_matrix):
        activated_input = self(input_matrix)
        return activated_input * (1 - activated_input)


class DenseBlock(object):

    def __init__(self, input_length=1, node_count=1):
        self._dense = Dense(
            node_count=node_count, input_length=input_length
        )
        self._activation = Sigmoid(input_length=input_length)

    @property
    def weights(self):
        return self._dense.weights

    def __call__(self, input_matrix):
        weighted_input = self._dense(input_matrix)
        return self._activation(weighted_input)

    def derived_activation_fn(self, input_matrix):
        weighted_input = self._dense(input_matrix)
        return self._activation.derived_fn(weighted_input)