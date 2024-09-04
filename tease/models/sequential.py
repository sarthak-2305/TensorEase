import numpy as np
from tease.losses.mse import MeanSquaredError

class Sequential:
    def __init__(self):
        self.sequence = []

    def add(self, layer):
        self.sequence.append(layer)

    def forward_pass(self, inputs):
        for layer in self.sequence:
            inputs = layer.forward(inputs)
            
        return inputs
    
    def backwards_pass(self, grad):
        for layer in self.sequence[::-1]:
            grad = layer.backward(grad)


    def optimizer(self):
        for layer in self.sequence:
            if layer.is_layer():
                layer.update_params(0.1)
    

    def train(self, X, y, epochs):
        self.pred_loss = None
        self.forward_outputs = None
        loss = MeanSquaredError()
        for i in range(epochs):
            print("epoch number", i)
            # print(self.forward_outputs)
            self.forward_outputs = self.forward_pass(X)

            self.pred_loss = loss.forward(self.forward_outputs, y)
            loss_grad = loss.backward()

            self.backwards_pass(loss_grad)
            self.optimizer()
            
    
    def result(self):
        print("The final outputs are:\n", self.forward_outputs)
        print("The final prediction loss is:\n", self.pred_loss)
        

    def predict(self, values):
        predictions = self.forward_pass(values)
        return predictions