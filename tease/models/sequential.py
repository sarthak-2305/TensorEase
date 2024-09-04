import numpy as np

class Sequential:
    def __init__(self):
        self.sequence = []

    def add(self, layer):
        self.sequence.add(layer)

    def forward(self, inputs):
        temp = None
        for i in self.sequence:
            inputs = i.forward(inputs)
        

    def train(self, epochs):
        for i in range(epochs):
            #forward
            #loss
            #gradients
            #optimize
            pass
