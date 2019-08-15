"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d, K = X.shape[0], X.shape[1], mixture.p.shape[0]
    # calculate Cu
    c_u = 1 - (X==0).astype(int) # n x d
    c_u_d = (np.sum(c_u, axis=1)).reshape(n,1) # n x 1
    X_mu = X.reshape(n, 1, d) - mixture.mu.reshape(1, K, d) # n x K x d
    X_mu = X_mu*c_u.reshape(n, 1, d)
    X_mu_square = np.sum(np.square(X_mu, dtype=np.float), axis=2)  # n x K
    X_mu_square_var = - X_mu_square/(2*mixture.var.reshape(1, K)) # n x K

    # For this function, you will want to use log(mixture.p[j] + 1e-16) instead of log(mixture.p[j]) to avoid numerical underflow
    # f(u,j) = log(pi) + log(N(X|theta))
    log_N = X_mu_square_var - (c_u_d/2)*np.log(2*np.pi*mixture.var.reshape(1, K)) # n x K
    f_u_i = np.log((mixture.p.reshape(1, K)+1e-16),) + log_N # n x K
    ll = logsumexp(f_u_i, axis=1) # n x 1
    ll = np.sum(ll) # scalar
    # l(j|u) = f(u,j) - log_sum(exp(f(u,j))
    # log_sum(exp(f(u,j)) = x_star + log_sum(x - x_star), x_star = max(x)
    f_u_i_star = np.max(f_u_i, axis=1) # n
    l_j_u = f_u_i - (f_u_i_star.reshape(n,1) + logsumexp(f_u_i - f_u_i_star.reshape(n, 1), axis=1).reshape(n,1)) # n x K

    p_j_u = np.exp(l_j_u)
    return p_j_u, ll



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d, k = X.shape[0], X.shape[1], mixture.var.shape[0]
    # hat_p = sum( p(j|u) )
    hat_p = np.sum(post, axis=0)/n # k

    # delta_l_c_u is indicator matrix of X which cell is not equal zero
    delta_l_c_u = 1 - (X==0).astype(int) # n x d

    # delta(l, c_u)*p(j|u) is a mechanism to leave mu of missing value cell unchanged
    delta_p = delta_l_c_u.reshape(n,1,d) * post.reshape(n,k,1) # n x k x d
    delta_p_sum_n = np.sum(delta_p, axis=0) # k x d

    row_mu = np.sum(delta_p * X.reshape(n,1,d), axis=0) # k x d
    hat_mu = (row_mu/np.sum(delta_p, axis=0))  # k x d

    much_less_mu_index = delta_p_sum_n<1
    hat_mu[much_less_mu_index] = mixture.mu[much_less_mu_index]


    # hat_sigma_square = (1/(sum( dim(u) * p(j|u)))) * sum( p(j|u) * norm(X - hat_mu)**2 )
    dim_u = np.sum(delta_l_c_u, axis=1) # n x 1
    dim_u_post = dim_u.reshape(n,1) * post # n x k
    x_error = (X.reshape(n, 1, d) - hat_mu.reshape(1, k, d))*delta_l_c_u.reshape(n,1,d) # n x k x d
    x_error_square = np.sum(np.square(x_error), axis=2) # n x k
    hat_sigma_square = (1/np.sum(dim_u_post, axis=0)) * np.sum(post * x_error_square, axis=0) # k
    hat_sigma_square[hat_sigma_square < min_variance] = min_variance
    mixture = GaussianMixture(hat_mu, hat_sigma_square, hat_p)

    return mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
