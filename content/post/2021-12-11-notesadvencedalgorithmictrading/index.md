---
title: NOTESAdvencedAlgorithmicTrading
author: Shiwz11
date: '2021-12-11'
slug: notesadvencedalgorithmictrading
categories:
  - NOTES
tags:
  - QUANT
---

# Bayesian Inference of a Binomial Proportion

## Bayesian Approach

1. **Assumptions**

2. **Prior Beliefs**

In this example, we use a relatively flexible probability distribution called the **beta distribution** to model our beliefs.

3. **Experimental Data**

4. **Posterior Beliefs**

5. **Inference**

**Assumptions of the Approach**:1.Our coin can only have two outcomes, that is, it can never land on its side;2.Each flip of the coin is completely independent of the others;3.The fairness of the coin does not change in time, which means that it is stationary.

## Bayes's Rule

$$
P(\theta|D) = P(D|\theta)P(\theta) / P(D)
$$

- $P(\theta)$: the piror.

- $P(\theta|D)$: posterior.

- $P(D|\theta)$: likelihood.

- $P(D)$: evidence

### Beta Distribution

PDF (probability density function) of the deta distribution:

$$
P(\theta|\alpha ,\beta ) = \theta ^{\beta -1} / B(\alpha, \beta)
$$

**共轭**：如果先验分布和似然函数可以使得先验分布和后验分布有相同的形式，那么就称先验分布和似然函数是共轭的，之所以如此，是可以使得现在的后验分布可以作为下一次计算的先验分布，就形成了一个链条。

$$
P(\theta |D) \propto P(D|\theta) P(\theta)
$$

后验分布正比于先验分布和似然函数的乘积。

> If our prior belief is specified by a beta distribution and we have a Bernoulli likelihood function, then our posterior will also be a beta distribution.

### Ways to specify a Beta Prior

If we can find a relationship between the mean, variance and the $\alpha$ , and $\beta$ , we can more easily specify our beliefs.

mean $\mu$ is given by:

$$
\mu = \frac{\alpha}{\alpha + \beta}
$$

standard deviation $\sigma$ is given by:

$$
\sigma = \sqrt{\frac{\alpha \beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}}
$$

$\alpha$ is given by:

$$
\alpha = \left( \frac{1-\mu}{\sigma^2} - \frac{1}{\mu} \right) \mu^2
$$

$\beta$ is given by:

$$
\beta = \alpha \left( \frac{1}{\mu} -1 \right)
$$

Be careful not to specify a $\sigma > 0.289$ cause this is the standard deviation of a uniform density which itself implies no prior belief on any particular fairness of the coin.

### Using Bayes' Rule to Calculate a Posterior

$$
\begin{aligned}
P(\theta \mid z, N) &=P(z, N \mid \theta) P(\theta) / P(z, N) \\
&=\theta^{z}(1-\theta)^{N-z} \theta^{\alpha-1}(1-\theta)^{\beta-1} /[B(\alpha, \beta) P(z, N)] \\
&=\theta^{z+\alpha-1}(1-\theta)^{N-z+\beta-1} / B(z+\alpha, N-z+\beta)
\end{aligned}
$$

In which, the denominator function $B(\dots)$ is the **Beta function**.

> If our prior is given by $beta(\theta|\alpha, \beta)$ and we observe $z$ heads in $N$ flips subsequently, then the posterior is given by $beta(\theta|z+\alpha, N-z +\beta)$.

# Markov Chain Monte Carlo

Markov Chain Monte Carlo (MCMC) is a way to calculate posterior distribution allows us to approximate the posterior distribution as calculated by Bayes' Theorem when there exists no conjugate priors.

在贝叶斯法则中，我们需要计算出分母$P(D)$:

$$
P(D) = \int_{\Theta} P(D,\theta)d\theta
$$

然而，在真实的建模过程中，我们通常难以找打上面模型的解析解，因此，需要MCMC求得其近似的数值解。

## The Metropolis Algorithm

Metropolis Algorithm is the basis of the MCMC family.

Basic recipes for most MCMC tend to follow this pattern:

1. Begin the algorithm at the current position in parameter space ($\theta_{current}$).

2. Propose a "Jump" to a new position in parameter space ($\theta_{new}$).

3. Accpet or reject the jump probabilistically using the prior information and available data.

4. If the jump is accepted, move to the new position and return to step 1.

5. If the jump is rejected, stay where you were and return to step 1.

6. After a set number of jumps have occurred, return all of the accepted positions.

The Metropolis algorithm uses a normal distribution to propose a jump. This normal distribution has a mean $\mu$ which equals to the current parameter position in parameter space and takes a "proposal width" for its standard deviation $\sigma$. 

The probability of moving:

$$
p = P(\theta_{new}) / P(\theta_{current})
$$

## Inferring a Binomial Proportion with Markov Chain Monte Carlo

```{python}
import matplotlib.pyplot as plt
import numpy as np
import pymc3

from scipy import stats
plt.style.use("ggplot")

# Parameter value for piror and analytic posterior
n = 50
z = 10
alpha = 12
beta = 12
alpha_post = 22
beta_post = 52

# How many iterations of the Metropolis
# algorithm to carry out for MCMC
iterations = 100000

# Use PyMC3 to construct a model context
basic_model = pymc3.Model()
with basic_model:
    # Define our prior belief about the fairness
    # of the coin using a Beta distribution
    theta = pymc3.Beta("theta", alpha=alpha, beta=beta)

    # Carry out the MCMC analysis using the Metropolis algorithm
    # Use Mximum A Posterori (MAP) optimisation as initial value for MCMC
    start = pymc3.find_MAP()

    # Use the Metropolis algorithm (as opposed to NUTS or HMC, etc.)
    step = pymc3.Metropolis()

    # Calculate the trace
    trace = pymc3.sample(
        iterations, step, start, random_seed=1, progressbar=True
    )

# Plot the posterior histogram from MCMC analysis
bins = 50
plt.hist(
    trace["theta"], bins,
    histtype="step",
    label="Posterior (MCMC)", color="red"
)

# Plot the analytic prior and posterior beta distributions
x = np.linspace(0, 1, 100)
plt.plot(
    x, stats.beta.pdf(x, alpha, beta),
    "--", label="Prior", color="blue"
)

plt.plot(
    x, stats.beta.pdf(x, alpha_post, beta_post),
    label="Posterior (Analytic)", color="green"
)

# Update the graph labels
plt.legend(title="Parameters", loc="best")
plt.xlabel("$\\theta$, Fairness")
plt.ylabel("Density")
plt.show()

# Show the trace plot
pymc3.traceplot(trace)
plt.show()
```

# Bayesian Linear Regression

## Frequentist Linear Regression

$$
f(X) = \beta_0 + \sum_{j=1}^{p} X_j \beta_j + \epsilon = \beta^{T}X + \epsilon
$$


$$
\begin{aligned}
\operatorname{RSS}(\beta) &=\sum_{i=1}^{N}\left(y_{i}-f\left(x_{i}\right)\right)^{2} \\
&=\sum_{i=1}^{N}\left(y_{i}-\beta^{T} x_{i}\right)^{2}
\end{aligned}
$$


$$
\hat{\beta} = (X^T X)^{-1}X^Ty
$$

## Bayesian Linear Regression

Acutually, Bayesian linear regression is stated in a probabilistic manner.

$$
\mathbf{y} \sim \mathcal{N}\left(\beta^{T} \mathbf{X}, \sigma^{2} \mathbf{I}\right)
$$

- Prior Distributions: If we have any prior knowledge about the parameters $\beta$ then we can choose prior distributions that reflect this. If not, we choose non-informative priors.

- Posterior Distributions: In the Bayesian formaulatoin, we receive an entire probability distribution that characterises our uncertainty on te different $\beta$ coefficients. Then, we can quantify our uncertainty in $\beta$ via the variance of this posterior distribution.

### GLM

GLMs allow for response variables that have error distributions other than the normal distribution. The linear model is related to the response y via a "link function" and is assumed to be generated via a statistical distribution from the exponential distribution family, which encompasses normal, gamma, beta, chi-squared, Bernoulli, Poisson and others. The mean of this distribution:

$$
\mathbb{E}(\mathbf{y})=\mu=g^{-1}(\mathbf{X} \beta)
$$

In which, g is the so-called link function.

Variance:

$$
Var(y) = V(\mathbb{E}(y)) = V(g^{-1}\mathbf{X}\beta)
$$

```{python}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

sns.set(style="darkgrid", palette="muted")

def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    """
    Simulate a random dataset using a noisy
    linear process.

    :param N: Number of data points to simulate
    :param beta_0: Intercept
    :param beta_1: Slope of univariate predictor, X
    :param eps_sigma_sq:
    :return:
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0
    df = pd.DataFrame(
        {"x":
         np.random.RandomState(42).choice(
             map(
                 lambda x: float(x)/100.0,
                 np.arange(N)
             ), N, replace=False
         )}
    )
    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of response based on 'x'
    eps_mean = 0.0
    df["y"] = beta_0 + beta_1*df["x"] + np.random.RandomState(42).normal(
        eps_mean, eps_sigma_sq, N
    )

    return df

def glm_mcmc_inference(df, iterations=5000):
    """
    Calculates the Markov Chain Monte Carlo trace of
    a Generalised Linear Model Bayesian linear regression
    model on supplied data.

    :param df: DataFrame containing the data
    :param iterations: Number of iterations to carry out MCMC for
    :return:
    """
    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        # We use a Normal distribution for the likelihood
        pm.glm.glm("y ~ x", df, family=pm.glm.families.Normal())

        # Use Maximum A Posteriori (MAP) optimisation
        # as initial value for MCMC
        start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        step = pm.NUTS()

        # Calculate the trace
        trace = pm.sample(
            iterations, step, start,
            random_seed=42, progressbar=True
        )

    return trace

if __name__ == "__main__":
    # These are our "true" parameters
    beta_0 = 1.0    # Intercept
    beta_1 = 2.0    # Slope

    # Simulate 100 data points, with a variance of 0.5
    N = 200
    eps_sigma_sq = 0.5

    # Simulate the "linear" data using the above parameters
    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)

    # Plot the data, and a frequentist linear regression fit
    # using the seaborn package
    sns.lmplot(x="x", y="y", data=df, size=10)
    plt.xlim(0.0, 1.0)

    trace = glm_mcmc_inference(df, iterations=5000)
    pm.traceplot(trace[500:])
    plt.show()

    # Plot a sample of posterior regression lines
    sns.lmplot(x="x", y="y", data=df, size=10, fit_reg=False)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 4.0)
    pm.glm.plot_posterior_predictive(trace, samples=100)
    x = np.linspace(0, 1, N)
    y = beta_0 + beta_1*x
    plt.plot(x, y, label="True Regression Line", lw=3., c="green")
    plt.legend(loc=0)
    plt.show()
```