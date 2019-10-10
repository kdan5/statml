import numpy as np
from sigmoid import *

# Cost function for logistic regression
def cross_entropy_loss(theta, X, y):
    m = X.shape[0]
    y = y.reshape((100,1))
    h = sigmoid(np.dot(X, theta))
    J = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / m

    return J

# Gradient of the loss function
def cross_entropy_gradient(theta, X, y):
    m = X.shape[0]
    y = y.reshape((100,1))
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(np.transpose(X), h - y) / m
    grad = np.ndarray.flatten(grad)

    return grad