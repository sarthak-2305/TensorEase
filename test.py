from tease.layers.linear import Linear
import numpy as np
from tease.datasets.spiral import spiral_data
from tease.activations.relu import Relu
from tease.activations.softmax import Softmax
from tease.losses.mse import MeanSquaredError

# def spiral_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points)  # radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y


# np.random.seed(0)
# X =     [[1, 2, 3, 2.5], 
#          [2, 5, 1, 2], 
#          [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(5, 3)


layer1 = Linear(2, 5)
activation1 = Relu()
activation2 = Softmax()
layer2 = Linear(5, 2)
loss = MeanSquaredError()

layer1_out = layer1.forward(X)
print(layer1_out)

print('\n\n Now this:\n\n')
act1_out = activation1.forward(layer1_out)
print(act1_out)

layer2_out = layer2.forward(layer1_out)
act2_out = activation1.forward(layer2_out)

print()
print(layer2_out)

soft_out = activation2.forward(act2_out)
print(soft_out)

print(X.shape)

print(loss.forward(soft_out, y))