# Brief Overview
The approach is to use linear models for feature selection and further experiment with a more complex algorithm (xgboost). For feature selection, we used genetic algorithm. Our results show that between 30 and 40 number of features provide the best generalization in linear model. Our experiments with xgboost models didn't generalize better than linear models. This is even after Bayesian hyper parameter optimization. The details on xgboost is provided at the end.

## Linear Model



**The Cauchy-Schwarz Inequality**
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$