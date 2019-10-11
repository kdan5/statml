import numpy as np
import matplotlib.pyplot as plt

# Plot 2-dimensional data that is split between two classes
def plot_class(X, y):
    # Identify the indices by their classification
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    # Plot the data
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='r')
    plt.scatter(X[neg, 0], X[neg, 1], marker='.', c='b')