import numpy as np
import sys


# backtracking line search
def backtracking_line_search(X, delta, G):
    alpha = 0.3
    beta = 0.5
    t = 1
    FX = - np.log((B - (A * X).sum(axis=1)).sum(axis=0))
    Y = X + t * delta
    FY = - np.log((B - (A * Y).sum(axis=1)).sum(axis=0))
    # Check that X is in the domain
    while np.isnan(FY) > 0 or \
          FY > FX + alpha * t * np.dot(G.T, delta):

        print FY.sum(), (FX + alpha * t * np.dot(G.T, delta)).sum()
        Y = X + t * delta
        FY = - np.log((B - (A * Y).sum(axis=1)).sum(axis=0))
        t = beta * t
    return t

m = 2
n = 2
# m * n matrice
A = np.random.randint(100, size=(m, n)) * np.random.random(size=(m, n))
B = np.random.randint(1000, size=(m, )).astype(float) * \
    np.random.random(size=(m, ))


def compute_gradient(X, B, A):
    # Compute gradient G
    det = (B - (A * X).sum(axis=1))
    det = det.repeat(n).reshape((m, n))
    a = A / det
    G = a.sum(axis=0)
    return G


def compute_hessian(X, B, A):
    # Compute gradient G
    det = (B - (A * X).sum(axis=1))
    det = det.repeat(n).reshape((m, n))
    a = A / det
    H = np.dot(a.T, a)
    return H


eps = 10e-10
t = 1
delta = np.zeros((n, ))
max_iter = 100
verbose = True

X = - np.random.random(size=(n, )).astype(float)

logfile = sys.stdout
# Keep track of all the computed values of X, in order to plot convergence.
values = []
FXs = []
for niter in range(max_iter):
    if verbose:
        logfile.write('\r %d %%' % (float(niter + 1) / max_iter * 100))
        logfile.flush()
    X += t * delta
    values.append(X.copy())
    FX = - np.log((B - (A * X).sum(axis=1)).sum(axis=0))
    FXs.append(FX)
    # Compute Newton's step
    G = compute_gradient(X, B, A)
    H = compute_hessian(X, B, A)

    # Compute Hessian H
    L = np.linalg.cholesky(H)
    L_inv = np.linalg.inv(L)

    delta = - np.dot(np.dot(L_inv.T, L_inv), G)
    lambda2 = (np.dot(L_inv, G) ** 2).sum()
    assert lambda2 > 0
    if lambda2 / 2 < eps:
        if verbose:
            print "Break out of the loop at %d" % niter
        break

    # TODO implement the backtracking line search to choose t
    t = backtracking_line_search(X.copy(), delta.copy(), G.copy())

# Now that we have an optimal X, let's compute the distance of values to X
X_opt = X.repeat(len(values))
X_opt = X_opt.reshape((n, len(values))).T
distances = ((values - X_opt) ** 2).sum(axis=1)

# TODO plot the convergence curve
