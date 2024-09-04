import numpy as np

class Sigmoid:
    def forward(self, X):
        self.inputs = X
        return 1/(1 + np.exp(-X))