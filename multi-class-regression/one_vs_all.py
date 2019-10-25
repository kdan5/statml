import numpy as np
import scipy.optimize as opt
from cross_entropy_loss_reg import *

'''
Compute one-vs-all logistic regression to perform a multi-class classification.
'''
def one_vs_all(X, y, reg_lambda, K):
    m, n = X.shape

    theta_mat = np.zeros((K, n + 1))

    X = np.concatenate((np.ones((m, 1)), X), -1)

    options = {'maxiter':400}
    
    for k in range(K):
        theta = np.zeros((n + 1, 1))
        y_k = (y == k).astype(int)
        out = opt.minimize(fun=cross_entropy_loss_reg, x0=theta, method='TNC', jac=cross_entropy_gradient_reg, options=options, args=(X, y_k, reg_lambda))
        theta_mat[k, :] = out.x
    return theta_mat

