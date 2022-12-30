---
title: NOTES-Asset Management A Systematic Approach to Factor Investing
author: Shiwz11
date: '2022-12-18'
slug: notes-asset-management-a-systematic-approach-to-factor-investing
categories:
  - NOTES
tags:
  - Factor Investment
---

Author:Andrew Ang 业界和学界通吃的大佬！

Asset management is actually about defining the bad times of investors, managing the bad times when investment factors do badly, and mitigating the bad times brought by delegated agents. Asset allocation, the practice of choosing appropriate asset classes and investments, is actually factor management.

# Asset Owners

# Preferences

Volatility strategy from the point of view of the investor selling volatility insurance.

## Risk

## Risk Aversion

Utilities convey the notion of "how you feel". The utility function defines bad times for the investors.

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221220145351.png)

- The utility curves exhibit diminishing marginal utility in wealth, which is an appealing property of concave utility functions.

- Bad times are times of high marginal utility.Good times are when marginal utility is low, which in Figure 2.3 correspond to when the asset owner is wealthy and the utility function is very flat.

- The investor's degree of risk aversion governs how painful bad times are for the asset owner.

![Concavity of Utility functions](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221220150235.png)

## How Risk Averse Are You

CRRA (constant relative risk aversion) utility takes the following form:

$$
U(W) = \frac{W^{1-\gamma}}{1-\gamma}
$$

- An attractive property of CRRA utility is that it leads to portfolio weights that do not depend on wealth, risk aversion is the same for all levels of wealth.

## Expected Utility

$$
U = E[U(W)] = \sum p_s U(W_s)
$$

To make choice, the asset owner maximizes expected utility. The problem is formally stated as:

$$
max_{\theta} E[U(W)]
$$
 
## What choice theory is not about

- Wealth

- Happiness

- Rationality versus Behavioral approaches

Most people do no have rational expected utility.

## The normative versus positive debate

The theory that we develop advocates:

- Diversifying widely

Many hold concentrated positions in their employers' stock; Many also suffer from home bias and thus fail to invest overseas.

- Rebalancing

- Dis-saving after retirement

- Using factors, not asset class labels, in investing

## Non-monetary Considerations

Like politics or so-called human-rights.

# Mean-variance Utility

Under mean-variance framework, the bad times are defined as follws: low means and high variance.

$$
U = E(r_p) - \frac{\gamma}{2} var(r_p)
$$

$$
E\left[U\left(1+r_p\right)\right] \approx U\left(1+E\left(r_p\right)\right)+\frac{1}{2} U^{\prime \prime}\left(1+E\left(r_p\right)\right) \operatorname{var}\left(r_p\right),
$$

# Realistic Utility Functions

Shortcomings of mean-variance utility:

- The variance treats the upside and downside the same (relative to the mean.)

- Only the first two moments matter

- Subjective probabilities matter

Actually, people tend to overestimate the probability of disasters.

- Bad times other than low means and high variance matter.

Actually, your utility function can be relative.

## Safety First

## Loss Aversion or Prospect Theory

Two parts about prospect theory:

1. Loss aversion utility

The pain of losses to be greater than the joy from gains. Loss aversion utility allows the investor to have different slopes for gains and losses.

2. Probability transformations

Decision weights allow investors to potentially severely overweight probability events--including both disasters and winning the lottery.

We can write prospect theory as a form of expected utility:

$$
U = \sum w(p_s)  U(W_s)
$$

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221221112159.png)

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221221112603.png)

## Disappointment Aversion

The asset owner cares more about downside versus upside outcomes.

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221221112908.png)

Investors overweight losses relative to gains, just like in loss aversion utility. What is different to loss aversion is that the **reference point** is endogenous.

## Habit Utility

Habit utility falls into a class of utility funcitons that define bad times not just by wealth outcomes, but also an **investor's environment.**

It is wealth **relative to a reference point that is important**.

- risk aversion is endogenous

- habit evolves over time

## Catching up with the Joneses

What matter is your performance relative to your peers.

## Uncertainty Aversion (Ambiguity Aversion)

*Knight uncertainty*

- [Ambiguity in asset pricing and portfolio choice: a review of the literature](https://link.springer.com/article/10.1007/s11238-012-9343-2)

- [Obervable Long-Run Ambiguity and Long-Run Risk](https://www.semanticscholar.org/paper/Observable-Long-Run-Ambiguity-and-Long-Run-Risk-Ulrich/df8dc1ddd88bde7108112a8e32feca223122d02e)
  Shows how the set of beliefs evolve over time and how they can be used to infer future asset prices.

## Picking Up Nickels and Dimes in Front of a Steamroller Redux

# Mean-Variance Investing

## Mean-Variance Frontiers

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221230095856.png)

## Diversification

Figure 3.3 plots the United States and Japan mean-variance frontier of different correlation values. As the correlation decreases, the tip of the mean-variance frontier pushes to the left.

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221230101555.png)

![](https://cdn.jsdelivr.net/gh/shiwz11/pics_public/20221230101733.png)








































