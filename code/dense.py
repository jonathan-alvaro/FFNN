import numpy as np


class Dense(object):
    def __init__(self, node_count: int = 1, input_length: int = 1):
        if input_length <= 0:
            raise ValueError(
                "Invalid input array length, expected at least 1"
            )
        self._input_length = input_length

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
    def input_length(self):
        return self._input_length

    @property
    def output_shape(self):
        return (1, self.node_count)

    @property
    def weights(self):
        return self._weights