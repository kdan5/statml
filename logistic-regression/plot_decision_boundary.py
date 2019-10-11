import numpy as np
import matplotlib.pyplot as plt
from map_feature import *

# Plot the decision boundary of the logistic regression algorithm
def plot_decision_boundary(theta, X, y):

    if (X.shape[1] <= 3):
        x_points = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        y_points = - (theta[0] + (theta[1] * x_points)) / theta[2]
        
        plt.plot(x_points, y_points)
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        for i in range(u):
            for j in range(v):
                z[i, j] = map_feature(u[i], v[j]) * theta

        plt.contour(u, v, z, [0,0], linewidth=2)




