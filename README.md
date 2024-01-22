## Brief Overview
The approach is to use linear models for feature selection and further experiment with a more complex algorithm (xgboost). For feature selection, we used genetic algorithm. Our results show that around 30 many features provide the best generalization in linear model. Our experiments with xgboost models didn't generalize better than linear models. The details on xgboost are provided at the end.

## Linear Model
We are minimizing MSE on a subset of data defined as $(X,y)$ with each time instance occupying a row in $X$ and $y$. 

<!-- We store the components of normal equations $X^TX$, $X^Ty$ and $y^Ty$ before running any optimization.  -->

$$\text{min}_{w} ||Xw - y||_2^2 + \lambda ||w||_2^2$$

Our feature selection algorithm learns feature selection mask $m$ and regularization parameter $\lambda$ jointly. For each $(m,\lambda)$ pair, we solve the normal equations on a train set and acquire an MSE value on the test set. Below use subscript to indicate the corresponding submatrix or vector.

$$(X_m^TX_m+\lambda I)w_{m}^* = X_m^Ty,\quad (X,y) \text{ train dataset}$$

$$\text{MSE}(m,\lambda) = (w_{m}^*)^TX^TX(w_{m}^*) - 2(w_{m}^*)^TX^Ty + y^Ty,\quad (X,y) \text{ test dataset}$$

$$\text{R}^{2}(X,y,w) = 1 - \text{MSE}(X,y,w)/||y||^2_2$$

