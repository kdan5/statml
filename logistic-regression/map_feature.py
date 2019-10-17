import numpy as np
'''
Maps features to polynomial features.
Returns a feature vector containing 1, X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, ...
'''
def map_feature(X1, X2):
    degree = 6
    out = np.ones(X1.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out