import numpy as np

'''
Compute the sigmoid function with the given input.
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))