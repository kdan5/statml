import numpy as np

# Cost function for linear regression algorithm
def computeCost(X, y, theta):

    J = 0
    m = len(y)

    h = np.dot(X, theta)
    J = np.sum(np.power(h - y, 2)) / (2 * m)
    
    return J
    
    
