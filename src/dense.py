import numpy as np
from activation import relu, drelu, sigmoid, dsigmoid, tanh, dtanh

class Dense:
    def __init__(self, input_size, output_size, activation):
        self.input = None 
        self.z = None
        
        self.input_size = input_size
        self.output_size = output_size

        self.activation = activation

        self.weights = np.random.rand(input_size, output_size)-0.5
        self.bias = np.random.rand(1, output_size)-0.5

    def forward(self, input):
        self.input = np.reshape(input, (1, -1))  ## Reshape to (1,...)
        self.z = np.dot(self.input, self.weights) + self.bias
        if self.activation == 'relu':
            return relu(self.z)
        elif self.activation == 'sigmoid':
            return sigmoid(self.z)
        elif self.activation == 'tanh':
            return tanh(self.z)
    
    def backward(self, dy, lr):
        if self.activation == 'relu':
            dZ = drelu(self.z)*dy
        elif self.activation == 'sigmoid':
            dZ = dsigmoid(self.z)*dy
        elif self.activation == 'tanh':
            dZ = dtanh(self.z)*dy

        dX = np.dot(dZ, self.weights.T)
        dW = np.dot(self.input.T, dZ)
        db = dZ

        # Gradient descent
        self.weights = self.weights - lr*dW
        self.bias = self.bias - lr*db 

        return dX