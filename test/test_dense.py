import unittest

import numpy as np

from code.dense import Dense


class TestDense(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_layer = Dense(node_count=10, input_length=100)

    def test_dense_init(self):
        default_layer = Dense()
        self.assertRaises(ValueError, Dense, input_length=0)
        self.assertRaises(ValueError, Dense, node_count=-1)

    def test_dense_node_count(self):
        one_node_layer = Dense(node_count=1)
        self.assertEqual(one_node_layer.node_count, 1)
        ten_node_layer = Dense(node_count=10)
        self.assertEqual(ten_node_layer.node_count, 10)

    def test_dense_input_length(self):
        short_input_layer = Dense(node_count=1, input_length=10)
        self.assertEqual(short_input_layer.input_length, 10)
        medium_input_layer = Dense(node_count=1, input_length=100)
        self.assertEqual(medium_input_layer.input_length, 100)

    def test_dense_output_shape(self):
        downscale_layer = Dense(node_count=10, input_length=100)
        self.assertEqual(downscale_layer.output_shape, (1, 10))
        upscale_layer = Dense(node_count=1000, input_length=100)
        self.assertEqual(upscale_layer.output_shape, (1, 1000))

    def test_dense_weights(self):
        self.assertEqual(self.test_layer.weights.shape, (100, 10))

    def test_call_layer(self):
        test_input = np.random.random_sample((10, 100))

        np.testing.assert_array_equal(
            self.test_layer(test_input),
            test_input @ self.test_layer.weights
        )

        self.assertRaises(
            ValueError,
            self.test_layer, np.array((1, 1))
        )