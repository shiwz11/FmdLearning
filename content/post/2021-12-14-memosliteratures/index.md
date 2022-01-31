---
title: MEMOSLiteratures
author: shiwz11
date: '2021-12-14'
slug: memosliteratures
categories:
  - Memos
tags:
  - QUANT
---


# 随手记

often factors are estimated as e.g. in the case of principal components (PCs) and factor mimicking portfolios (albeit the latter is not needed in our setting). This generates an additional layer of uncertainty normally ignored in empirical analysis due to the associated asymptotic complexities. Nevertheless, it is relatively easy to adjust the Bayesian estimators of risk premia to account for this uncertainty. In the case of a mimicking portfolio, under a diffuse prior and Normal errors, the posterior distribution of the portfolio weights follow the standard Normal-inverse-Gamma of Gaussian linear regression models (see e.g. Lancaster(2004)). Similarly, in the case of principal components as factors, under a diffuse prior, the covariance matrix from which the PCs are constructed follow an inverse-Wishart distribution. Hence, the posterior distributions in Definitions 1 and 2 can account for the generated factors uncertainty by the Normal-inverse-Gamma posterior of the mimicking portfolios coeffcients, and then sampling the remaing parameters as explained in the definitions.

# Big Data

## Ecominics/Predictions/Sparsity

consider a linear model to predict a response variable $y_t$:

$$
y_t = u_{t}^{\prime} \phi + x_{t}^{\prime} \beta +\varepsilon_{t}
$$


