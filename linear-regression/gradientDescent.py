import numpy as np
from computeCost import *

# Gradient descent to find theta
def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))

    for i in range(iterations):
        h = np.dot(X, theta)
        # Fix line: output is (97, 2) should be (2, 1)
        theta = (theta - np.transpose(alpha * np.dot(np.transpose(h - y), X)) / m) 
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
