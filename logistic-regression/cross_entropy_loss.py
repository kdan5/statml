import numpy as np
from sigmoid import *

# Cost function for logistic regression
def cross_entropy_loss(X, y, theta):

    h = sigmoid(np.dot(X, theta))
    J = - np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    grad = np.dot(np.transpose(X), h - y)

    return J, grad