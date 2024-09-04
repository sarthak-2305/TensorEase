import numpy as np

class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 
    
        return self.dinputs
    
    def is_layer(self):
        return False
    

class LeakyRelu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs > 0, dvalues, self.alpha * dvalues)
        return self.dinputs

    def is_layer(self):
        return False
    