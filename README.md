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

The cross-validation is done by splitting the data into 8 approximately equal parts in time, i.e. a 1-7 split in test-train. The data is not shuffled before the split in order to avoid mixing any non-stationary nature of the data. The loss function is defined by averaging MSE on the test sets across all 8 splits. Below we index each partition with $i$.

$$
\text{loss-fun}(f) =\big(\sum_{i = 0}^{n - 1} \text{MSE}(m,\lambda;i)\big)/\big(\sum_{i = 0}^{n - 1} \text{MSE}(0,\lambda;i)\big) + 
\epsilon (||m||_1 + \lambda)
$$

In the loss function the MSE values are normalized with $\sum_i\text{MSE}(0,\lambda;i)$ which is a constant equal to $y^Ty$, the total energy across the whole target values. The second part of the loss function is to promote reduction in number of parameters and L2-regularization. The value of $\epsilon$ is chosen as $10^{-4}$ so that this part only runs in effect when MSE is no longer decreasing significantly.

We use $R^2$ value from cross-valiation as the primary metric to understand how well the calibrations generalize. $R^2$ defined by one minus the first part of the loss-fun.

$$
\text{R}^2(f) = 1 - \big(\sum_{i = 0}^{n - 1} \text{MSE}(m,\lambda;i)\big)/\big(\sum_{i = 0}^{n - 1} \text{MSE}(0,\lambda;i)\big)
$$

## Data Pre-Processing

We create different calibrations for different parts of the day. For this we experimented with 2 and 4 split of the day into equal parts. In our experiments, we see that 4 way split didn't create good generalization for second and third parts. We decided to use 2 way split for the whole project. 

We get rid of rows with Q1,Q2 < 0.9999. We set any X and Y value with larger than 999 absolute value to NaN. Any row with a NaN on X or Y is removed. This causes a 7% reduction in data size. 

## Feature Selection

The features are nearly degenerate. In order to get rid of those which are linearly dependent on others, we run a QR-decomposition on $X^TX$ coming from the first training set. The zeros on R matrix diagonals emerge when prior columns of cancel out that column. We get rid of features that correspond to a diagonal value amplitude less than 1e-5 (relative to the largest diagonal value). This reduces the number of features from 375 to 233. We continue our feature selection process with the remaining oness. 

We use the library [geneticalgorithm](https://github.com/rmsolgi/geneticalgorithm) in order to minimize the loss function. The algorithm starts with randomly a generated population of 1000 individuals. Each individual is a vector in the form of $(m,\log_{10}(\lambda))$. $m$ as an integer takes values from $\{0,1\}$ and $\log_{10}(\lambda) \in [-12,-1]$ is a real number. In each round random pairs are crossed to create 2 offsprings by splitting the sequence of numbers $(m,\log_{10}(\lambda))$ on a random point and crossing the values. At each round individual are selected according to their fitness and with a chance of $1/300$, an entry in an individual gets mutated. We stop the algorithm after 200 generations pass with no improvements on the best score. 

The genetic algorithm results are not necessarily optimal. Sometimes, we can improve it through a greedy algorithm towards less number of features and less regularization. In this method, at each step we either change a 1 to 0 in $m$ or reduce $\log_{10}(\lambda)$ by $0.5$. Among all possible decisions, we take the one that creates the largest drop in the loss. The iterations continue until no more any loss reduction is possible. In our test runs greedy algorithm improving the genetic results. But with the chosen small mutation rate and population size, we do not observe any step taken by the greedy algorithm.

## Results





\% $R^2$ | Morning | Afternoon | 
--- | --- | --- | 
Y1 | 0.357 | 283 |
Y2 | 3.189e-03 | |

