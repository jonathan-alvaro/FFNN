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
    @abstractmethod
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
        self._current_output = []

    def _verify_input_shape(self, input_matrix):
        matrix_shape = input_matrix.shape

        if matrix_shape[-1] != self.input_length:
            raise ValueError(
                "Expected {} features, got {} instead".format(
                    self.input_length, matrix_shape[-1]
                )
            )

    def _sigmoid(self, z, derivative=False):
        if derivative:
            return np.multiply(z, (1-z))
        else:
            return 1 / (1 + np.exp(-z))

    def __call__(self, input_matrix):
        self._verify_input_shape(input_matrix)
        self._current_output = np.matmul(input_matrix, self.weights)
        for i, row in enumerate(self._current_output):
            for j, val in enumerate(row):
                self._current_output[i, j] = self._sigmoid(
                    self._current_output[i, j])
        return self._current_output

    def _gradient(self, X, y):
        pred = self(X)
        grad = X.T.dot(pred - y)
        return grad

    def backpropagate(self, prev_error):
        gradient = self._gradient()
        delta = self._momentum * gradient + self._learning_rate * gradient

        next_error = delta.dot(self._weights.T)
        self._weights = self._weights + delta
        return next_error

    @property
    def node_count(self):
        return self._node_count

    @property
    def output_shape(self):
        return 1, self.node_count

    @property
    def weights(self):
        return self._weights
