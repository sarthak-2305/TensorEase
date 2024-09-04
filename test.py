from tease.layers.linear import Linear
import numpy as np
from tease.datasets.spiral import Spiral
from tease.activations.relu import Relu
from tease.activations.softmax import Softmax
from tease.losses.mse import MeanSquaredError
from tease.datasets.nonlinear import NonLinearData


np.random.seed(42)
data = NonLinearData(8, 1)
X, y = data.load_data()
# data.show()

print(X.shape)
print(y.shape)

layer1 = Linear(1, 5)
relu = Relu()
layer2 = Linear(5, 1)
loss = MeanSquaredError()

layer1_out = layer1.forward(X)
relu_out = relu.forward(layer1_out)
layer2_out = layer2.forward(relu_out)

pred_loss = loss.forward(layer2_out, y)


print(X)
print()
print(y)
print()
print(layer2_out)
print()
print(pred_loss)
print()


d_grad_loss = loss.backward()
d_layer2_out = layer2.backward(d_grad_loss)
d_relu = relu.backward(d_layer2_out)
d_layer1_out = layer1.backward(d_relu)

print('d_layer_out')
print(d_layer1_out)
print()

layer1.update_params(0.1)
layer2.update_params(0.1)

layer1_out = layer1.forward(X)
relu_out = relu.forward(layer1_out)
layer2_out = layer2.forward(relu_out)

pred_loss = loss.forward(layer2_out, y)

print(X)
print()
print(y)
print()
print(layer2_out)
print()
print(pred_loss)
print()



d_grad_loss = loss.backward()
d_layer2_out = layer2.backward(d_grad_loss)
d_relu = relu.backward(d_layer2_out)
d_layer1_out = layer1.backward(d_relu)

print('d_layer_out')
print(d_layer1_out)
print()

layer1.update_params(0.1)
layer2.update_params(0.1)

layer1_out = layer1.forward(X)
relu_out = relu.forward(layer1_out)
layer2_out = layer2.forward(relu_out)

pred_loss = loss.forward(layer2_out, y)

print(X)
print()
print(y)
print()
print(layer2_out)
print()
print(pred_loss)
print()


d_grad_loss = loss.backward()
d_layer2_out = layer2.backward(d_grad_loss)
d_relu = relu.backward(d_layer2_out)
d_layer1_out = layer1.backward(d_relu)

print('d_layer_out')
print(d_layer1_out)
print()

layer1.update_params(0.1)
layer2.update_params(0.1)

layer1_out = layer1.forward(X)
relu_out = relu.forward(layer1_out)
layer2_out = layer2.forward(relu_out)

pred_loss = loss.forward(layer2_out, y)

print(X)
print()
print(y)
print()
print(layer2_out)
print()
print(pred_loss)
print()