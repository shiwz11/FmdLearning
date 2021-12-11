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