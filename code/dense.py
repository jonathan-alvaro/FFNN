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
        self._weights = np.ndarray(
            (self.input_length, self.node_count), dtype=np.float32
        )

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