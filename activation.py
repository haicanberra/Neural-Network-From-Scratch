import numpy as np

def relu(input):
    return np.maximum(0, input)
    
def drelu(z):
    dz = np.copy(z)
    dz[z<0] = 0
    dz[z>=0] = 1
    return dz

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def dsigmoid(input):
    return np.multiply(sigmoid(input),(1 - sigmoid(input)))

def tanh(input):
    return np.tanh(input)

def dtanh(input):
    return 1-np.tanh(input)**2