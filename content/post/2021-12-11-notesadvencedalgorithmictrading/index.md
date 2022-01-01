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

# Time Series

> Time series analysis attempts to understand the past and predit the future.

- Trends

- Seasonal Variation

- Series Dependence

## Serial Correlation

## Random Walks and White Noise Models

## ARMA

## ARIMA-GARCH

## Cointegrated Time Series

# State Space Models and the Kalman Filter

Different from the ARIMA model, the parameters of state space models can adapt over time.

The goal of the state space model is to infer information about the states, given the observation, as new information arrives.

There are three types of inference that are of interest when considering state space models:

- Prediction

- Filtering

- Smoothing

## Linear State-Space Model

$\theta_{t}$: column vector of the stats;

$y_t$: observation of the model at time t;

$G_t$: state-transition matrix betweet current and prior states at tiem t and t-1 respectively;

$v_t$: measurement noise;

$w_t$: system noise drawn from a multivariate normal distribution;

$F_{t}^{T}$: linear dependece matrix of $\theta_t$ on $y_t$.

$m_0$: mean value of the multivariate normal distribution of the initial state, $\theta_0$;

$C_0$: variance-covariance matrix of the multivariate normal distribution of the initial state, $\theta_0$; 

$W_t$: the variance-covariance matrix for the multivariate normal distribution from which the system noise is drawn;

$V_t$: variance-covariance matrix for the multivariate normal distribution from which the measurement noise is drawn.

We assume that these states are linear combination of the prior state at time $t-1$ as well as system noise (random variation), which is drawn from a multivariate normal distribution.

**State equation:**

$$
\theta_t = G_t \theta_{t-1} + w_t
$$

**Observation equation:**

$$
y_t = F_{t}^{T} \theta_t + v_t
$$


$$
\begin{aligned}
\theta_{0} & \sim \mathcal{N}\left(m_{0}, C_{0}\right) \\
v_{t} & \sim \mathcal{N}\left(0, V_{t}\right) \\
w_{t} & \sim \mathcal{N}\left(0, W_{t}\right)
\end{aligned}
$$

## Kalman Filter

Apply Bayes' Rule:

$$
P(\theta_t |D_{t-1},y_t) = \frac{P(y_t|\theta_t)P(\theta_t|D_{t-1})}{P(y_t)}
$$

Means we can update our view on the state, $\theta_t$, in a rational manner given the fact that we have new information in the form of the current observation $y_t$.


**prior**:

$$
\theta_t | D_{t-1} \sim \mathcal{N}(a_t, R_t)
$$

**likelihood**:

$$
y_t | \theta_{t} \sim \mathcal{N}(F_{t}^{T}\theta_t, V_T)
$$

**posterior**:

$$
\theta_t|D_{t} \sim \mathcal{N}(m_t, C_t)
$$

Next, this is how Kalman Filter links all of the terms above together:

$$
\begin{aligned}
a_{t} &=G_{t} m_{t-1} \\
R_{t} &=G_{t} C_{t-1} G_{t}^{T}+W_{t} \\
e_{t} &=y_{t}-f_{t} \\
m_{t} &=a_{t}+A_{t} e_{t} \\
f_{t} &=F_{t}^{T} a_{t} \\
Q_{t} &=F_{t}^{T} R_{t} F_{t}+V_{t} \\
A_{t} &=R_{t} F_{t} Q_{t}^{-1} \\
C_{t} &=R_{t}-A_{t} Q_{t} A_{t}^{T}
\end{aligned}
$$

```{python}
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as np
import pandas as pd
import pandas_datareader as pdr
from pykalman import KalmanFilter

def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the
    changing relationship between the sets of prices
    :param etfs:
    :param prices:
    :return:
    """
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates latter dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlOrRd')
    colours = np.linspace(0.1, 1, plen)

    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=30, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8
    )

    # Add a colour bar for the date colouring and set the
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.show()

def calc_slope_intercept_kalman(etfs, prices):
    """
    Utilise the Kalman Filter from the PyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    :param etfs:
    :param prices:
    :return:
    """
    delta = le-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2,2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )

    state_means, state_covs = kf.filter(prices[etfs[1]].values)
    return state_means, state_covs

def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept changes from the
    Kalman Filter calculated values.
    :param prices:
    :param state_means:
    :return:
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0],
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()

if __name__ == "__main__":
    # Choose the ETF symbols to work along with
    # start and end dates for the price histories
    etfs = ['TLT', 'IEI']
    start_date = "2010-08-01"
    end_date = "2016-08-01"

    # Obtain the adjusted closing prices from Yahoo finance
    etf_df1 = pdr.get_data_yahoo(etfs[0], start_date, end_date)
    etf_df2 = pdr.get_data_yahoo(etfs[1], start_date, end_date)
    prices = pd.DataFrame(index=etf_df1.index)
    prices[etfs[0]] = etf_df1["Adj Close"]
    prices[etfs[1]] = etf_df2["Adj Close"]

    draw_date_coloured_scatterplot(etfs, prices)
    state_means, state_covs = calc_slope_intercept_kalman(etfs, prices)
    draw_slope_intercept_changes(prices, state_means)
```

# Hidden Markov Models

Used for Market Regime detection.

A stochastic state space model involves random transitions between states where the probability of the jump is only dependent upon the current state, rather than any of the previous states. The model is said to possess the Markov Property and is thus "memoryless".

|            | Fully Observable        | Partially Observable                         |
| ---------- | ----------------------- | -------------------------------------------- |
| Autonomous | Markov Chain            | Hidden Markov Model                          |
| Controlled | Markov Decision Process | Partially Observable Markov Decision Process |

In a HMM, there are underlying latent states-and probability transitions between them, but they are not directly observable.

$$
\begin{aligned}
p\left(X_{1: T}\right) &=p\left(X_{1}\right) p\left(X_{2} \mid X_{1}\right) p\left(X_{3} \mid X_{2}\right) \ldots \\
&=p\left(X_{1}\right) \prod_{t=2}^{T} p\left(X_{t} \mid X_{t-1}\right)
\end{aligned}
$$

This states that the probability of seeing sequence of observations is given by the probability of the initial observation multiplied $T-1$ tiems by the conditional probability of seeing the subsequent observation (transition function), which it self is time-independent.

To simulate n steps of a DSCM (Discrete-State Markov Chain) model we define the n-step transition matrix $A(n)$ as:

$$
A_{ij}(n) := p(X_{t+n} = j|X_t=i) 
$$

For HMM, it is necesary to create a set of discrete states $z_t \in \{1, \dots, K\}$ and to model the observations with an additional probability model, $p(\mathbf{x}_{t}|z_t)$. Which means, the conditional probability of seeing a particular obseration (asset return) given that the state (market regime) is currently equal to $z_t$.

### HMM mathematical specification

Murphy (2012) gives the following joint density function for the HMM.

$$
\begin{aligned}
p\left(\mathbf{z}_{1: T} \mid \mathbf{x}_{1: T}\right) &=p\left(\mathbf{z}_{1: T}\right) p\left(\mathbf{x}_{1: T} \mid \mathbf{z}_{1: T}\right) \\
&=\left[p\left(z_{1}\right) \prod_{t=2}^{T} p\left(z_{t} \mid z_{t-1}\right)\right]\left[\prod_{t=1}^{T} p\left(\mathbf{x}_{t} \mid z_{t}\right)\right]
\end{aligned}
$$

The model choice of the obsevation transition function is more complex, which in general, we use a confitional multivariate Gaussiann distribution with mean $\mu_k$ and covariance $\sigma_k$:

$$
p\left(\mathbf{x}_{t} \mid z_{t}=k, \theta\right)=\mathcal{N}\left(\mathbf{x}_{t} \mid \mu_{k}, \sigma_{k}\right)
$$

Which means, if the state $z_t$ is currently equal to k, then the probability of seeing observation $x_t$, given the parameters of the model $\theta$, is distributed as a multivariate Gaussian.

## Regime Detection with HMM

Regime detection is actually a form of **unsupervised learning**.


### Simulated Data

```{r}
if (!require(depmixS4)) {
  install.packages('depmixS4')
  require('depmixS4')
}

if (!require('quantmod')) {
  install.packages('quantmode')
  require('quantmod')
}

set.seed(1)

# The bull market is distributed as N(0.1,0.1)
# the bear market is distributed as N(-0.05, 0.2)
# Create the parameters for the bull and
# bear market returns distributions
Nk_lower = 50
Nk_upper = 150
bull_mean = 0.1
bull_var = 0.1
bear_mean = -0.05
bear_var = 0.2

# Create the list of durations (in days) for each regime
days = replicate(5, sample(Nk_lower:Nk_upper, 1))

# Create the various bull and bear markets returns
market_bull_1 = rnorm(days[1], bull_mean, bull_var)
market_bear_2 = rnorm(days[2], bear_mean, bear_var)
market_bull_3 = rnorm(days[3], bull_mean, bull_var)
market_bear_4 = rnorm(days[4], bear_mean, bear_var)
market_bull_5 = rnorm(days[5], bull_mean, bull_var)

# Create the list of true regime states and full returns list
# 1 for bullish 2 for bearish
true_regimes = c(rep(1, days[1]), rep(2, days[2]), rep(1, days[3]),
                 rep(2, days[4]), rep(1, days[5]))
returns = c(market_bull_1, market_bear_2, market_bull_3,
            market_bear_4, market_bull_5)

plot(returns, type = 'l', xlab = '', ylab = 'Returns')

# Create and fit the HMM

hmm = depmixS4::depmix(returns ~ 1, family = gaussian(), nstates = 2, 
                       data = data.frame(returns = returns))
hmmfit = fit(hmm, verbose = F)

# Output both the true regimes and the 
# posterior probabilities of the reigmes
post_probs = depmixS4::posterior(hmmfit)
layout(1:2)
plot(post_probs$state, type = 's', main='True Regimes', 
     xlab='', ylab='Regime')

matplot(post_probs[,-1], type='l', main='Regime Posterior Probabilities', 
        ylab='Probability')
legend(x='topright', c('Bull', 'Bear'), fill=1:2, bty='n')
```

### Financial Data

Downlaod the data using the quantmod library.

```{r}
# Obtain SP500 data from 2004 onward and 
# create the reutrns stream from this
quantmod::getSymbols("^GSPC", from='2004-01-01')
gspcRets = diff(log(Cl(GSPC)))
returns = as.numeric(gspcRets)
plot(gspcRets)

# Fit a Hidden Markov Model with two states
# to the SP500 returns stream
hmm = depmix(returns ~ 1, family=gaussian(), nstates=2,
             data=data.frame(returns=returns))
hmmfit = fit(hmm, verbose=F)
post_probs = posterior(hmmfit)

# Plot the returns stream and the posterior
# probabilities of the separate regimes
layout(1:2)
plot(returns, type='l', main='Regime Detection', xlab='', ylab='Returns')
matplot(post_probs[,-1], type='l', 
        main='Regime Posterior Probabilities', 
        ylab='Probability')

legend(x='bottomleft', c('Regime #1', 'Regime #2'), fill=1:2, bty='n')

# Fit a Hidden Markov Model with three states
# to the SP500 returns stream
hmm = depmix(returns ~ 1, family=gaussian(), nstates=3, 
             data=data.frame(returns=returns))
             
hmmfit = fit(hmm, verbose=F)
post_probs = posterior(hmmfit)

# Plot the returns stream and the posterior
# probabilities of the separate regimes
layout(1:2)
plot(returns, type='l', main='Regime Detection', 
     xlab='', ylab='Returns')
     
matplot(post_probs[,-1], type='l', 
        main='Regime Posterior Probabilities', 
        ylab='Probability')
legend(x='bottomleft', c('Regime #1', 'Regime #2', 'Regime #3'), fill=1:3, bty='n')
```

# Introduction to Machine Learning

Algorithms frequently used within quantitative finance:

- Linear Regression

- Linear Classification

- Tree-Based Methods

- SVM

- Artificial Neural Networks and Deep Learning

- Bayesian Networks

- Clustering

- Dimensionality Reduction
