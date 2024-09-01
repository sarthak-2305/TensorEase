import numpy as np

# np.random.seed(0)
# X =     [[1, 2, 3, 2.5], 
#          [2, 5, 1, 2], 
#          [-1.5, 2.7, 3.3, -0.8]]

class Linear:
    def __init__(self, input_dim, neurons):
        self.weights = np.random.randn(input_dim, neurons) * 0.1
        self.bias = np.zeros((1, neurons))

        self.dW = None
        self.dW = None

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
    
    def backward(self, d_out):
        pass

