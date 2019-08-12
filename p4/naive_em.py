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
    # probability of p(x|theta)
    n , d, K = X.shape[0], X.shape[1], mixture.mu.shape[0]
    XX = X.reshape(n, 1, d)
    mu = mixture.mu.reshape(1, K, d)
    prior = np.exp(-np.square(XX - mixture.mu)/(2 * mixture.var))
    raise NotImplementedError


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
    raise NotImplementedError
