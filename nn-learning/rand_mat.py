import numpy as np

def rand_mat(rows, cols):

    w = np.zeros((rows, cols))
    epsilon = 0.12

    w = (np.random.rand(rows, cols) * 2 * epsilon) - epsilon

    return w