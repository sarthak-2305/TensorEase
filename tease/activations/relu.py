import numpy as np

class Relu:
    def forward(self, X):
        self.inputs = X
        return np.maximum(0, X)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 
    