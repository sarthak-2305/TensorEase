import numpy as np

# np.random.seed(0)
# X =     [[1, 2, 3, 2.5], 
#          [2, 5, 1, 2], 
#          [-1.5, 2.7, 3.3, -0.8]]

class Linear:
    def __init__(self, input_dim, neurons):
        self.weights = np.random.randn(input_dim, neurons) * 0.1
        self.biases = np.zeros((1, neurons))


    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs
    
    def update_params(self, alpha):
        self.weights -= alpha * self.dweights
        self.biases -= alpha * self.dbiases