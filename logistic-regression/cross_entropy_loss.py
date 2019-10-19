import numpy as np
from sigmoid import *

'''
Compute the cross entropy loss function.
'''
def cross_entropy_loss(theta, X, y):
    m, n = X.shape
    y = y.reshape((m, 1))
    theta = theta.reshape((n, 1))
    h = sigmoid(np.dot(X, theta))
    J = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m

    return J


'''
Compute the gradient for the cross entropy loss function.
'''
def cross_entropy_gradient(theta, X, y):
    m, n = X.shape
    y = y.reshape((m, 1))
    theta = theta.reshape((n, 1))
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(np.transpose(X), h - y) / m
    grad = np.ndarray.flatten(grad)

    return grad