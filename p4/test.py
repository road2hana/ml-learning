import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here

mu = np.array([[2., 4., 5., 5., 0.],[3., 5., 0., 4., 3.], [2., 5., 4., 4., 2.], [0., 5., 3., 3., 3.]])

var = np.array([5.93, 4.87, 3.99, 4.51])
pi = np.array([0.25, 0.25, 0.25, 0.25])

mixture = common.GaussianMixture(mu, var, pi)
post, ll = em.estep(X, mixture)

print(post)
print(ll)

mixture = em.mstep(X, post, mixture)
print(mixture)
