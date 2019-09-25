import numpy as np
import matplotlib.pyplot as plt
from plot_class import *
from map_feature import *

def plot_decision_boundary(X, y, theta):
    plot_class(X[:,1:2], y)

    if (X.size[1] <= 3):
        x_points = [np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2]
        y_points = (-1 / theta[2]) * (theta[1] * x_points) + theta[0]

        plt.plot(x_points, y_points)

        plt.legend('Admitted', 'Not Admitted', 'Decision Boundary')
        plt.axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        for i in range(u):
            for j in range(v):
                z[i, j] = map_feature(u[i], v[j]) * theta

        plt.contour(u, v, z, [0,0], linewidth=2)




