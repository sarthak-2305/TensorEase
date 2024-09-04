import numpy as np
import matplotlib.pyplot as plt

class Spiral:
    def __init__(self, points, classes):
        self.X = np.zeros((points*classes, 2))
        self.y = np.zeros(points*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(points*class_number, points*(class_number+1))
            r = np.linspace(0.0, 1, points)  # radius
            t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
            self.X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            self.y[ix] = class_number
        
    def show(self):
        X = self.X
        y = self.y
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def load_data(self):
        return self.X, self.y


