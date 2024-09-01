class Relu:
    def forward(self, x):
        self.input = x
        return max(0, x)
    
    def backward(self, y):
        inp = y
        inp[self.input <= 0] = 0
        return inp