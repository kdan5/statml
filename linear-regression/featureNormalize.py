import numpy as np

def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
        mu[0,i] = np.mean(X[:,i])
        sigma[0,i] = np.std(X[:,i])

    X_norm = np.divide(np.subtract(X_norm, mu), sigma)

    return X_norm, mu, sigma
