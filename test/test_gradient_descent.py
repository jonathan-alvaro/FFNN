import unittest

import numpy as np

from code.optimizer import GradientDescentOptimizer


class TestGradientDescent(unittest.TestCase):
    def setUp(self) -> None:
        self.test_optimizer = GradientDescentOptimizer()

    def test_init(self):
        test_optimizer = GradientDescentOptimizer()

    def test_learning_rate(self):
        self.assertEqual(self.test_optimizer.learning_rate, 0.01)

        self.test_optimizer.learning_rate = 0.1
        self.assertEqual(self.test_optimizer.learning_rate, 0.1)

