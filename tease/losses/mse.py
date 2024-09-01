import numpy as np

class MeanSquaredError:
    def forward(self, pred, actual):
        self.loss = np.mean((pred.T - actual) ** 2)

        self.pred = pred
        self.actual = actual

        return self.loss
    
    def backward(self):
        N = self.pred.shape[0]

        dpred = (2/N) * (self.pred - self.actual)

        return dpred