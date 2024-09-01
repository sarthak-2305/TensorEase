import numpy as np

class Relu:
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    
    def backward(self, Y):
        inp = Y
        inp[self.input <= 0] = 0
        return inp