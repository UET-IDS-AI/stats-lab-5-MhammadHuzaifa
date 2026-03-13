import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2) / (2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    prior_A = 0.3
    prior_B = 0.7

    likelihood_A = gaussian_pdf(time, 40, 4)
    likelihood_B = gaussian_pdf(time, 45, 4)

    numerator = likelihood_B * prior_B
    denominator = likelihood_A * prior_A + likelihood_B * prior_B

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    prior_A = 0.3
    prior_B = 0.7

    # generate class labels
    labels = np.random.choice(["A", "B"], size=n, p=[prior_A, prior_B])

    # generate samples
    samples = np.zeros(n)

    samples[labels == "A"] = np.random.normal(40, 4, np.sum(labels == "A"))
    samples[labels == "B"] = np.random.normal(45, 4, np.sum(labels == "B"))

    # check samples close to given time
    tolerance = 0.5
    mask = np.abs(samples - time) < tolerance

    if np.sum(mask) == 0:
        return 0

    B_count = np.sum(labels[mask] == "B")

    return B_count / np.sum(mask)
