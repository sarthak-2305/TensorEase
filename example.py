import numpy as np
import matplotlib.pyplot as plt
from tease.layers.linear import Linear
from tease.activations.relu import Relu, LeakyRelu
from tease.datasets.nonlinear import NonLinearData
from tease.models.sequential import Sequential
# from tease.activations.sigma import Sigmoid

# np.random.seed(0)
data = NonLinearData(1000, 1)
X, y = data.load_data()
X_train, X_test, y_train, y_test = data.split_data()
# data.show()

# print(X)
# print()
# print(y)
# print()


model = Sequential()
model.add(Linear(1, 32))
model.add(LeakyRelu())
model.add(Linear(32, 32))
model.add(LeakyRelu())
# model.add(Linear(32, 16))
# model.add(LeakyRelu())
model.add(Linear(32, 1))

model.train(X_train, y_train, 500)
model.result()

# new_data = np.array([5])
# answer = model.predict(new_data)
# print("Prediction is: ", answer)

predictions = model.predict(X_test)


plt.scatter(X, y, color='blue', alpha=0.6, label='True Data')
plt.title("Synthetic Dataset with Model Predictions")
plt.xlabel("X")
plt.ylabel("y")


plt.scatter(X_test, predictions, color='red', label='Model Predictions')

plt.legend()
plt.show()