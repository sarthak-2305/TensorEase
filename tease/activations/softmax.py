import numpy as np


class Softmax:
    def forward(self, X):
        values = np.exp(X)
        return values / np.sum(values, axis=1, keepdims=True)