class GradientDescentOptimizer(object):
    def __init__(self, learning_rate=0.01):
        self._lr = learning_rate

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, new_learning_rate):
        self._lr = new_learning_rate

    def optimize_model(self, model):
        layers = model.layers

        errors = []

        for i, layer in reversed(enumerate(layers)):
            if i == len(layers) - 1:
                layer_error =