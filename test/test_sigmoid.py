import unittest

import numpy as np

from code.layers import Sigmoid

class TestSigmoid(unittest.TestCase):
    def test_init(self):
        test_sigmoid = Sigmoid()

    def test_call(self):
        test_layer = Sigmoid(input_length=100)
        test_input = np.random.random_sample((10, 100))

        expected_output = 1 / (1 + np.exp(-test_input))
        np.testing.assert_array_equal(
            test_layer(test_input), expected_output
        )

    def test_derived_fn(self):
        test_layer = Sigmoid(input_length=1)
        test_input = np.random.random_sample((1000, 1))

        activated_input = test_layer(test_input)
        expected_output = activated_input * (1 - activated_input)
        output = test_layer.derived_fn(test_input)
