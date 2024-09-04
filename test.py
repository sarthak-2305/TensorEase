from tease.layers.linear import Linear
import numpy as np
from tease.datasets.spiral import Spiral
from tease.activations.relu import Relu
from tease.activations.softmax import Softmax
from tease.losses.mse import MeanSquaredError
from tease.datasets.nonlinear import NonLinearData


# np.random.seed(0)
# X =     [[1, 2, 3, 2.5], 
#          [2, 5, 1, 2], 
#          [-1.5, 2.7, 3.3, -0.8]]

# data = Spiral(100, 3)
# X, y = data.load_data()
# data.show()

data = NonLinearData(20)
X, y = data.load_data()
data.show()

# print(X.shape)
# print(y.shape)
# print(X)
# print(y)

layer1 = Linear(1, 5)
activation1 = Relu()
layer2 = Linear(5, 1)
# activation2 = Softmax()
loss = MeanSquaredError()

layer1_out = layer1.forward(X)
# print(layer1_out)

# print('\n\n Now this:\n\n')
act1_out = activation1.forward(layer1_out)
# print(act1_out)

layer2_out = layer2.forward(act1_out)
# act2_out = activation1.forward(layer2_out)

print()
# print(layer2_out)

# soft_out = activation2.forward(act2_out)
print(layer2_out)


print(loss.forward(layer2_out, y))