import numpy as np

'''
Compute the sigmoid function with the given input.
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
Compute the gradient of the sigmoid function.
'''
def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))