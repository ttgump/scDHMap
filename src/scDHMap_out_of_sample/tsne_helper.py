import torch
import torch.nn.functional as F

import numpy as np
import numba
from sklearn.metrics import pairwise as sklearn

MAX_EXP = np.log(np.finfo(np.float32).max)

@numba.njit(fastmath=True)
def gaussian_kernel_entropy(D, beta):
    """
    Computes the Gaussian kernel values for a vector of distances,
    and the entropy of the distribution.
    """

    # Compute P-row and corresponding entropy
    P = -D * beta
    shift = MAX_EXP - P.max()
    P = np.exp(P + shift)
    P_sum = np.sum(P)
    H = np.log(P_sum) - shift + beta * np.sum(D * P) / P_sum
    return H, P / P_sum


def transition_probabilities(D, perplexity, tol=1e-5, max_iter=50):
    """
    Computes the transition probabilites of a distance matrix D.
    Performs a binary search to get P-values in such a way
    that each conditional Gaussian has the same perplexity.
    Identifies appropriate sigmas to get nearest neighbors
    up to some tolerance and calculates the required
    precision (1 / variance^2) to obtain a Gaussian
    kernel with the desired perplexity.

    Parameters:
    -----------
    D : array_like, shape (n_samples, n_samples)
        The distance matrix for calculating probabilities
    perplexity : float, default = 30.0
        The desired uncertainty.
        This indirectly specifies the number of nearest neighbors.
    tol : float, default = 1e-5
        The desired perplexity is obtained up to some tolerance
    max_iter : int, default = 50
        The maximum number of iterations for calculating the
        required precision.

    Returns:
    --------
    P : ndarray, shape (n_samples, n_samples)
        The final Gaussian kernel
    beta : ndarray
        The employed precisions per instance.
    """

    # Initialize variables
    n_samples = D.shape[0]
    P = np.zeros((n_samples, n_samples))
    beta = np.ones(n_samples)  # empty precision vector
    entropy = np.log(perplexity) / np.log(2.0)

    for idx in numba.prange(n_samples):

        # Set minimum and maximum values for precision
        betamin = -np.inf
        betamax = np.inf

        # Compute the Gaussian kernel and entropy for the current precision
        D_idx = D[idx]
        D_idx = np.concatenate((D_idx[:idx], D_idx[idx + 1 :]))
        H, P_idx = gaussian_kernel_entropy(D_idx, beta[idx])

        # Evaluate whether the perplexity is within tolerance...
        H_diff = H - entropy
        n_iter = 0
        while np.abs(H_diff) > tol and n_iter < max_iter:
            # ...if not, increase or decrease precision
            if H_diff > 0:
                betamin = beta[idx]
                if np.isinf(betamax):
                    beta[idx] *= 2
                else:
                    beta[idx] = (beta[idx] + betamax) / 2
            else:
                betamax = beta[idx]
                if np.isinf(betamin):
                    beta[idx] /= 2
                else:
                    beta[idx] = (beta[idx] + betamin) / 2

            # Recompute the values
            H, P_idx = gaussian_kernel_entropy(D_idx, beta[idx])
            H_diff = H - entropy
            n_iter += 1
        P_idx = np.concatenate((P_idx[:idx], np.array([0.0]), P_idx[idx:]))

        # Set the final row of P
        P[idx] = P_idx

    return P


jit = numba.njit(fastmath=True, parallel=True)
transition_probabilities_parallel = jit(transition_probabilities)

def compute_gaussian_perplexity(D, perplexities, symmetric=True, parallel=True, square=True):
    multiscale = (len(perplexities) > 1)
    if square:
        D = D ** 2

    if parallel:
        if multiscale:
            P = [
                transition_probabilities_parallel(D, perplexity=perplexity)
                for perplexity in perplexities
            ]
            P = np.stack(P)
        else:
            P = transition_probabilities_parallel(
                D, perplexity=perplexities
            )
    else:
        if multiscale:
            P = [
                transition_probabilities(D, perplexity=perplexity)
                for perplexity in perplexities
            ]
            P = np.stack(P)
        else:
            P = transition_probabilities(
                D, perplexity=perplexities
            )

    if multiscale:
        P = P.mean(0)

    if symmetric:
        P = P + P.T

#    return P / np.sum(P)
    return P / np.sum(P, axis=1)