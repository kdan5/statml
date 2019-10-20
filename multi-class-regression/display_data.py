import numpy as np

def display_data(X, width):
    m, n = X.shape

    if width == None:
        width = round(np.sqrt(n))

    height = n / width
    rows = floor(np.sqrt(m))
    cols = ceil(m / rows)
    pad = 1

    display_array = - np.ones((rows * (height + pad) + pad, cols * (width + pad) + pad))

    curr = 0
    for i in range(rows):
        for j in range(cols):
            if curr > m:
                break
            max_val = max(abs(X[curr, :]))
            display_array[(i - 1) * (height + pad) + (range(height)) + pad, (j - 1) * (width + pad) + (range(width)) + pad] = X[curr, :].reshape((height, width)) / max_val
            curr = curr + 1
            if curr > m:
                break
    
    h = 
    return h, display_data