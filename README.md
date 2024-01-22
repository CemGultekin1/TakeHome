## Brief Overview
The approach is to use linear models for feature selection and further experiment with a more complex algorithm (xgboost). For feature selection, we used genetic algorithm. Our results show that around 30 many features provide the best generalization in linear model. Our experiments with xgboost models didn't generalize better than linear models. The details on xgboost are provided at the end.

## Linear Model
We are minimizing MSE on a subset of data defined as $(X,y)$ with each time instance occupying a row in $X$ and $y$. 



$$\text{min}_{w} ||Xw - y||_2^2 + \lambda ||w||_2^2$$

Our feature selection algorithm learns feature selection mask $m$ and regularization parameter $\lambda$ jointly. For each $f = (m,\lambda)$ pair, we solve the normal equations on a train set and acquire an MSE (mean squared error) value on the test set. Below use subscript to indicate the corresponding submatrix or vector.

$$(X_m^TX_m+\lambda I)w_{f} = X_m^Ty,\quad (X,y) = D_{\text{train}}$$

$$\text{MSE}(f) = w_{f}^TX_m^TX_mw_{f} -  2w_{f}^TX_m^Ty + y^Ty,\quad (X,y) = D_{\text{test}}$$

The cross-validation is done by splitting the data into 8 approximately equal parts in time, i.e. a 1-7 split in test-train. The data is not shuffled before the split in order to avoid mixing any non-stationary nature of the data. The cost function is defined by averaging MSE on the test sets across all 8 splits. 

$$
\text{cost-fun}(f) =\frac{\sum_{i = 0}^{n - 1} \text{MSE}_i(m,\lambda)}{\sum_{i = 0}^{n - 1} \text{MSE}_i(0,\lambda)} + 
\epsilon (||m||_1 + \lambda)
$$

In the cost-fun the MSE values are normalized with $\sum_i\text{MSE}_i(0,\lambda)$ which is a constant equal to $y^Ty$, the total energy across the whole target values.



