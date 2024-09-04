import numpy as np
import matplotlib.pyplot as plt

class NonLinearData:
    def __init__(self, num_samples=100, num_features=1, noise_level=0.1):
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise_level = noise_level
        
        # Generate the data when the class is instantiated
        self.X = None
        self.y = None
        self._generate_data()

    def _generate_data(self):
        # Generate random input features
        self.X = np.random.randn(self.num_samples, self.num_features)

        # Create a nonlinear relationship for the target variable
        # Example: y = sin(X) + X^2 + noise
        self.y = (self.X ** 2).sum(axis=1, keepdims=True) + \
                 (self.X ** 3).sum(axis=1, keepdims=True) + \
                 np.random.randn(self.num_samples, 1) * self.noise_level

    def show(self):
        if self.num_features == 1:
            plt.scatter(self.X, self.y, color='blue', alpha=0.6)
            plt.title("Synthetic Dataset")
            plt.xlabel("X")
            plt.ylabel("y")
            plt.show()
        else:
            # For multiple features, just plot the first feature against y
            plt.scatter(self.X[:, 0], self.y, color='blue', alpha=0.6)
            plt.title("Synthetic Dataset (Feature 1 vs y)")
            plt.xlabel("X[:, 0]")
            plt.ylabel("y")
            plt.show()

    def load_data(self):
        return self.X, self.y

    def split_data(self, train_ratio=0.8):
        # Split the data into training and testing sets
        split_index = int(train_ratio * self.num_samples)
        X_train, X_test = self.X[:split_index], self.X[split_index:]
        y_train, y_test = self.y[:split_index], self.y[split_index:]
        return X_train, X_test, y_train, y_test