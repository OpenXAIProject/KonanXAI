import numpy as np

v = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2] * 3

a = np.array(v)
print(a)

a = np.reshape(a, (-1, 8))
print(a, a.mean(1), a.shape)