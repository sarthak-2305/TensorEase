import numpy as np

class Sigmoid:
    def forward(self, X):
        self.inputs = X
        self.output = 1/(1 + np.exp(-X))
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues * ((self.output) * (1 - self.output))
    
        return self.dinputs
    
    def is_layer(self):
        return False