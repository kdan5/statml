import numpy as np
from sigmoid import *

'''
Compute the loss function for regularized cross entropy.
'''
def cross_entropy_loss_reg(theta, X, y, reg_lambda):
    m, n = X.shape
    y = y.reshape((m, 1))
    theta = theta.reshape((n, 1))
    h = sigmoid(np.dot(X, theta))
    J = (2 * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) + reg_lambda * np.sum(theta ** 2)) / (2 * m)

    return J


'''
Compute the gradient for the regularized cross entropy function.
'''
def cross_entropy_gradient_reg(theta, X, y, reg_lambda):
    m, n = X.shape
    y = y.reshape((m, 1))
    theta = theta.reshape((n, 1))
    h = sigmoid(np.dot(X, theta))
    grad = (np.dot(np.transpose(X), h - y))
    grad[1:] = grad[1:] + (reg_lambda / m) * theta[1:]
    grad = np.ndarray.flatten(grad)

    return grad
