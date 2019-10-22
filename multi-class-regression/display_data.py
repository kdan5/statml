import numpy as np
import matplotlib.pyplot as plt

'''
Reconstitutes and displays a set of 2D images in a square grid.
Parameters: X: The flattened image data matrix.
            width: The width of each image.
Returns:    h: The plotted image data.
            display_mat: The raw reconstitued image data in a matrix.
'''
def display_data(X, width=None):
    m, n = X.shape

    if width == None:
        width = round(np.sqrt(n))

    width = width.astype(np.int64)
    height = (n / width).astype(np.int64)

    rows = np.floor(np.sqrt(m)).astype(np.int64)
    cols = np.ceil(m / rows).astype(np.int64)

    pad = 1

    display_mat = - np.ones((pad + rows * (height + pad), pad + cols * (width + pad)))
    curr = 0
    for i in range(rows):
        for j in range(cols):
            if curr >= m:
                break
            max_val = max(abs(X[curr, :]))
            h_step = pad + i * (height + pad)
            w_step = pad + j * (width + pad)
            display_mat[h_step:(h_step + height), w_step:(w_step + width)] = np.reshape(X[curr, :], (height, width), -1) / max_val
            curr = curr + 1
            if curr >= m:
                break
    
    h = plt.imshow(display_mat)

    return h, display_mat