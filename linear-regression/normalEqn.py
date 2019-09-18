import numpy as np
import numpy.linalg as lg

def normalEqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    theta = np.dot(np.dot(lg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)

    return theta
