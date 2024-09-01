from tease.layers.linear import Linear
import numpy as np


np.random.seed(0)
X =     [[1, 2, 3, 2.5], 
         [2, 5, 1, 2], 
         [-1.5, 2.7, 3.3, -0.8]]

layer1 = Linear(4, 5)
layer2 = Linear(5, 2)

layer1_out = layer1.forward(X)
print(layer1_out)
layer2_out = layer2.forward(layer1_out)
print()
print(layer2_out)