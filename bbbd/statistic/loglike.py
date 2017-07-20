import numpy as np
from scipy.special import gammaln


def logfactorial(n):

    return gammaln(n + 1)

def regularized_log(vector):
    """
    A function which is log(vector) where vector > 0, and zero otherwise.

    :param vector:
    :return:
    """

    out = np.zeros_like(vector)
    idx = vector > 0
    out[idx] = np.log(vector[idx])

    return out


def xlogy(x, y):
    """
    A function which is 0 if x is 0, and x * log(y) otherwise. This is to fix the fact that for a machine
    0 * log(inf) is nan, instead of 0.

    :param x:
    :param y:
    :return:
    """

    return np.where(x > 0, x * regularized_log(y), 0)


def poisson_log_likelihood_no_bkg(observed_counts, expected_model_counts):
    """
    Poisson log-likelihood for the case where the background has no uncertainties:

    L = \sum_{i=0}^{N}~o_i~\log{(m_i)} - m_i - \log{o_i!}

    :param observed_counts:
    :param expected_model_counts:
    :return: (log_like vector, background vector)
    """

    # Model predicted counts
    # In this likelihood the background becomes part of the model, which means that
    # the uncertainty in the background is completely neglected

    log_likes = xlogy(observed_counts, expected_model_counts) - expected_model_counts - \
                logfactorial(observed_counts)

    return np.sum(log_likes)
