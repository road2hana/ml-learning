"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n , d, K = X.shape[0], X.shape[1], mixture.mu.shape[0]
    xx = X.reshape(n, 1, d)
    mu = mixture.mu.reshape(1, K, d)
    xx_mu_2 = np.sum(-np.square(xx - mu), axis=2) # n x K
    prior = np.exp(xx_mu_2/(2 * mixture.var)) # n x K
    prior = prior/np.power(2 * np.pi * mixture.var, d/2) # n x K
    prior = prior * mixture.p
    prob_sumKmodels = np.sum(prior, axis=1).reshape(n,1) # n x 1
    log_likelihood = np.log(prob_sumKmodels) # n x 1
    log_likelihood = np.sum(log_likelihood, dtype=np.float) # scalar
    post = prior/prob_sumKmodels
    log_post = np.log(post, dtype=np.float)
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K , d = post.shape[0], post.shape[1], X.shape[1]
    # hat_n_k = sum( p(k | i) )
    hat_n_k = np.sum(post, axis=0) # 1 x K
    hat_n_k = hat_n_k.reshape(K, 1) # K x 1

    # hat_mix_p = hat_n_k/n
    hat_mix_p = hat_n_k/n  # K x 1
    hat_mix_p = hat_mix_p.reshape(K)

    # hat_mu_k = (1/hat_n_k) * sum( p(k|i)*x_i )
    hat_mu_k = post.T.dot(X)/hat_n_k # K x d

    # hat_var = (1/(hat_n_k * d)) * sum(p(k|i) *  (x_i - hat_mu_k)**2 )
    xx_mu_2 = np.square((X.reshape(n, 1, d) - hat_mu_k.reshape(1, K, d)), dtype=np.float) # n x K x d
    xx_mu_2 = np.sum(xx_mu_2, axis=2, dtype=np.float) # n x K
    prob_xx_mu_2 = post * xx_mu_2 # n x K
    # hat_var_k = (1/(hat_n_k*d)) * sum(post * (xx - mu)**2 )
    hat_var_k = np.sum(prob_xx_mu_2, axis=0, dtype=np.float)/(d * hat_n_k.reshape(1, K)) # 1 x K
    hat_var_k = hat_var_k.reshape(K)

    mixture = GaussianMixture(hat_mu_k, hat_var_k, hat_mix_p)
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
    pre_cost = None
    cost = None
    while pre_cost is None or cost - pre_cost > 1e-6 * np.abs(cost):
        pre_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost
