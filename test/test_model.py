import unittest

import numpy as np

from code.layers import Dense
from code.model import Model


class TestModel(unittest.TestCase):
    def test_init(self):
        test_model = Model()

    def test_append_layer(self):
        test_model = Model()
        test_model.append_layer(Dense(node_count=10, input_length=100))
        self.assertEqual(test_model.depth, 1)

        self.assertRaises(
            ValueError,
            test_model.append_layer, Dense(node_count=1, input_length=5)
        )

    def test_call(self):
        test_model = Model()
        test_model.append_layer(Dense(node_count=10, input_length=100))
        test_model.append_layer(Dense(node_count=1, input_length=10))

        test_input = np.random.random_sample((4, 100))
        model_weights = test_model.weights
        expected_output = test_input
        for layer_weight in model_weights:
            expected_output = np.matmul(expected_output, layer_weight)

        np.testing.assert_array_equal(
            test_model(test_input), expected_output
        )

    def test_model_input_length(self):
        test_model = Model()
        self.assertEqual(test_model.input_length, 0)

        test_model.append_layer(Dense(
            node_count=100, input_length=1000
        ))
        self.assertEqual(test_model.input_length, 1000)

    def test_cost_fn(self):
        test_model = Model()
        test_model.append_layer(Dense(
            node_count=1, input_length=1000
        ))

        test_input = np.random.random_sample((1000, 1000))
        test_label = np.random.random_sample((1000, 1))
        output = test_model(test_input)
        expected_cost = np.mean(np.power(output - test_label, 2) / 2)
        cost = np.mean(test_model.cost_fn(test_input, test_label))
        self.assertEqual(cost, expected_cost)