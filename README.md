## Brief Overview
The approach is to use linear models for feature selection and further experiment with a more complex algorithm (xgboost). For feature selection, we used genetic algorithm. Our results show that around 30 many features provide the best generalization in linear model. Our experiments with xgboost models didn't generalize better than linear models. The details on xgboost are provided at the end.

## Linear Model

$$ \text{min}_{w} ||Xw - y||_2^2 + \lambda ||w||_2^2 $$
$$ X^TXw^{*} = X^Ty$$
$$ \text{MSE}(X,y,w) = w^TX^TXw - 2w^TX^Ty + y^Ty $$
$$ \text{R}^{2}(X,y,w) = 1 - \text{MSE}(X,y,w)/||y||^2_2 $$
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$