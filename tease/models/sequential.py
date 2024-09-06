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
                layer.update_params(0.01)
    

    def train(self, X, y, epochs, Xt=None, yt=None):
        # self.pred_loss = None
        self.forward_outputs = None
        loss = MeanSquaredError()

        history = {'loss': [], 'val_loss': []}
        for i in range(epochs):
            print("epoch number", i + 1)
            # print("loss:", self.pred_loss)
            print()
            # print(self.forward_outputs)
            self.forward_outputs = self.forward_pass(X)

            pred_loss = loss.forward(self.forward_outputs, y)
            history['loss'].append(pred_loss)

            loss_grad = loss.backward()

            self.backwards_pass(loss_grad)
            self.optimizer()

            if Xt is not None and yt is not None:
                self.forward_outputs_valid = self.forward_pass(Xt)
                val_loss = loss.forward(self.forward_outputs_valid, yt)
                history['val_loss'].append(val_loss)


        return history
            
    
    def result(self):
        print("The final outputs are:\n", self.forward_outputs)
        print("The final prediction loss is:\n", self.pred_loss)
        

    def predict(self, values):
        predictions = self.forward_pass(values)
        return predictions