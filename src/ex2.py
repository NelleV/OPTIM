import numpy as np
import sys
import cvxpy

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
max_iter = 10
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

# Let's compare the results with the one obtained with cvxpy
#x = cvxpy.variable(n,)
#p = cvxpy.program(cvxpy.minimize(np.dot(c.T, x)),
#                  [cvxpy.leq(np.dot(A, x).sum(axis=1), B)])
