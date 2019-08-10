import numpy as np

a = np.random.rand(10, 4)

a[:, 3] = a[:, 3] > 0.5

print(a)
ts = a[np.where(a[:, -1] == 0), :-1].T

tc = a[np.where(a[:, -1] == 1), :-1].T       