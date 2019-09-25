import numpy as np

def map_feature(X1, X2):
    degree = 6
    s = (X1[:, 1].shape[0], np.sum(range(8)))
    output = [1] * s

    for i in range(1, degree):
        for j in range(i):
            output.append(np.power(X1, i - j) * np.power(X2, j))
    
    return output