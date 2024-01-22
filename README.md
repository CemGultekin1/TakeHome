## Brief Overview
The approach is to use linear models for feature selection and further experiment with a more complex algorithm (xgboost). For feature selection, we used genetic algorithm. Our results show that around 30 many features provide the best generalization in linear model. Our experiments with xgboost models didn't generalize better than linear models. The details on xgboost are provided at the end.

## Linear Model
We are minimizing MSE on a subset of data defined as $(X,y)$ with each time instance occupying a row in $X$ and $y$. 



$$\text{min}_{w} ||Xw - y||_2^2 + \lambda ||w||_2^2$$

Our feature selection algorithm learns feature selection mask $m$ and regularization parameter $\lambda$ jointly. For each $f = (m,\lambda)$ pair, we solve the normal equations on a train set. Below use subscript to indicate the corresponding submatrix or vector. In order to keep $\lambda$ scale invariant we multiply it with the maximum entry of $X^TX$ on the training set.

$$(X_m^TX_m+s_X\lambda I)w_{f} = X_m^Ty$$

$$s_X =\max_{i,j} (X^TX)_{i,j}$$

$$(X,y) = D_{\text{train}}$$

Then we acquire an MSE (mean squared error) value on the test set. 

$$\text{MSE}(f) = w_{f}^TX_m^TX_mw_{f} -  2w_{f}^TX_m^Ty + y^Ty,\quad (X,y) = D_{\text{test}}$$

The cross-validation is done by splitting the data into 8 approximately equal parts in time, i.e. a 1-7 split in test-train. The data is not shuffled before the split in order to avoid mixing any non-stationary nature of the data. The cost function is defined by averaging MSE on the test sets across all 8 splits. Below we index each partition with $i$.

$$
\text{cost-fun}(f) =\big(\sum_{i = 0}^{n - 1} \text{MSE}(m,\lambda;i)\big)/\big(\sum_{i = 0}^{n - 1} \text{MSE}(0,\lambda;i)\big) + 
\epsilon (||m||_1 + \lambda)
$$

In the cost function the MSE values are normalized with $\sum_i\text{MSE}(0,\lambda;i)$ which is a constant equal to $y^Ty$, the total energy across the whole target values. The second part of the cost function is to promote reduction in number of parameters and L2-regularization. The value of $\epsilon$ is chosen as $10^{-4}$ so that this part only runs in effect when MSE is no longer decreasing significantly.

## Genetic Algorithm

The cost function is minimized with [geneticalgorithm](https://github.com/rmsolgi/geneticalgorithm). 



