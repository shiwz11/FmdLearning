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


