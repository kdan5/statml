import numpy as np
import matplotlib.pyplot as plt

def plot_class(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.scatter(X[pos, 0], X[pos, 1], marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], marker='.')
    plt.show()