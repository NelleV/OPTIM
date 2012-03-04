import numpy as np
import sys
from ex1 import newton_method

m = 2
n = 2
t = 1
# m * n matrice
A = np.random.randint(100, size=(m, n)) * np.random.random(size=(m, n))
B = np.random.randint(1000, size=(m, )).astype(float) * \
    np.random.random(size=(m, ))
c = np.random.random(size=(m, ))

verbose = True
mu = 1.5
eps = 10e-10
m = 1
max_iter = 300
values = []
logfile = sys.stdout
for it in range(max_iter):
    logfile.write('\r %s iterations' % it)
    logfile.flush()
    X, _, _ = newton_method(A, B, c, T=t)
    values.append(X)
    if m / t < eps:
        print "break at iteration %s" % it
        break
    t = mu * t
