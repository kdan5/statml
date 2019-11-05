import numpy as np
from sigmoid import *

def nn_cross_entropy(params, input_layer_size, hidden_layer_size, labels, X, y, Lambda):
    m, n = X.shape
    theta1 = params['theta1']
    theta2 = params['theta2']

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    a1 = np.concatenate((np.ones((m, 1)), X), -1)
    z2 = np.dot(a1, np.transpose(theta1))
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(z2)), -1)
    z3 = np.dot(a2, np.transpose(theta2))
    a3 = sigmoid(z3)

    y_mat = np.zeros((m, labels))

    for i in range(m):
        y_mat[i, y[i]] = 1

    loss = np.sum(np.sum((-y_mat * np.log(a3)) - ((1 - y_mat) * np.log(1 - a3)))) / m
    print("loss:", loss)
    reg_term = (np.sum(np.sum(np.power(theta1[:, 1:], 2))) + np.sum(np.sum(np.power(theta2[:, 1:], 2)))) * Lambda / m
    J = loss + reg_term

    for t in range(m):
        a_1 = np.concatenate((np.ones((1, 1)), X[t, :].reshape(n, 1)))
        z_2 = np.dot(theta1, a_1)
        a_2 = sigmoid(z_2)
        a_2 = np.concatenate((np.ones((1, 1)), a_2), 0)
        z_3 = np.dot(theta2, a_2)
        a_3 = sigmoid(z_3)

        y_log = np.transpose(range(labels) == y[t]).reshape((labels, 1))
        del_3 = a_3 - y_log
        del_2 = np.dot(np.transpose(theta2), del_3) * np.concatenate((np.ones((1, 1)), sigmoid_grad(z_2)), 0)
        del_2 = del_2[1:]
        theta1_grad = theta1_grad + del_2 * np.transpose(a_1)
        theta2_grad = theta2_grad + del_3 * np.transpose(a_2)

        p_prime = {'theta1_grad':theta1_grad, 'theta2_grad':theta2_grad}

    return J