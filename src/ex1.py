import numpy as np

m = 10
n = 7
# m * n matrice
A = np.random.randint(1000, size=(m, n)).astype(float)
B = np.random.randint(1000, size=(m, 1)).astype(float)

eps = 10e-10
t = 1
delta = np.zeros((n, ))
max_iter = 300
verbose = True

# Choose random point in dom F
X = np.random.randint(100, size=(n, )).astype(float)

# Keep track of all the computed values of X, in order to plot convergence.
values = []
for niter in range(max_iter):
    X += t * delta
    values.append(X)
    B = B.repeat(n, axis=1)
    # Compute Newton's step

    # Compute gradient G
    G = (1. / (B - A * X)).sum(axis=0)

    # Compute Hessian H
    # FIXME that is so wrong...
    H = np.identity(n)
    H *= (- A / (B - A * X) ** 2).sum(axis=0)

    delta = - np.dot(np.linalg.inv(H), G)
    lambda2 = np.dot(np.dot(G.T, np.linalg.inv(H)), G)

    if lambda2 / 2 < eps:
        if verbose:
            print "Break out of the loop at %d" % niter
        break

    # TODO implement the backtracking line search to choose t

# Now that we have an optimal X, let's compute the distance of values to X
values = np.array(values)
X_opt = X.repeat(len(values))
distances = ((values - X_opt) ** 2).sum(axis=1)

# TODO plot the convergence curve

# backtracking line search
alpha = 0.25
beta = 0.5
t = 1
Y = X + t * delta
F = np.log((B - A * X).sum(axis=0))
