import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tease.models.sequential import Sequential
from tease.layers.linear import Linear
from tease.activations.relu import Relu

np.random.seed(42)

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)


df_train = pd.DataFrame(X_train, columns=housing.feature_names)
print(df_train.head())

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.fit_transform(X_test)

y_train = y_train.reshape(-1, 1)
y_train.shape

model = Sequential()
model.add(Linear(X_train.shape[1], 30))
model.add(Relu())
model.add(Linear(30, 10))
model.add(Relu())
model.add(Linear(10, 1))

history = model.train(X_train, y_train, 20, X_valid, y_valid)

predictions = model.predict(X_test)


# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()