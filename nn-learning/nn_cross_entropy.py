import numpy as np
from sigmoid import *

def nn_cross_entropy_loss(params, input_layer_size, hidden_layer_size, labels, X, y, Lambda):
    m, n = X.shape
    theta1 = params['Theta1']
    theta2 = params['Theta2']

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    a1 = np.concatenate((np.ones((m, 1)), X), -1)
    z2 = np.dot(a1, np.transpose(theta1))
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(z2), -1))
    z3 = np.dot(a2, np.transpose(theta2))
    a3 = sigmoid(z3)

    y_mat = np.zeros((m, labels))

    for i in range(m):
        y_mat[i, y[i]] = 1

    loss = -np.sum(np.sum(y_mat * np.log(a3) + (1 - y_mat) * np.log(1 - a3))) / m
    reg_term = (np.sum(np.sum(np.power(theta1[:, 1:], 2))) + np.sum(np.sum(np.power(theta2[:, 1:], 2)))) * Lambda / m
    J = loss + reg_term

    return J