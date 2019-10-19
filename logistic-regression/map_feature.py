import numpy as np

'''
Maps features to polynomial features
Returns a feature vector containing 1, X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, ... up to the desired degree
degree: the maximum degree of the mapping
'''
def map_feature(X1, X2, degree=1):
    out = np.ones((X1.size, 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out
