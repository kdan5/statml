import numpy as np

'''
Maps features to polynomial features
Returns a feature vector containing the polynomial terms up to the given degree of X1 and X2
'''
def map_feature(X1, X2, degree=1):
    out = np.ones(X1.size).reshape((-1, 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j)).reshape((-1, 1))))
    return out
