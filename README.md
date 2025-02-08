[![Documentation Status](https://readthedocs.org/projects/sklearn-ensemble-cv/badge/?version=latest)](https://sklearn-ensemble-cv.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/sklearn_ensemble_cv?label=pypi)](https://pypi.org/project/sklearn-ensemble-cv)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/sklearn_ensemble_cv)](https://pepy.tech/project/sklearn_ensemble_cv)

# Ensemble Cross Validation


`sklearn_ensemble_cv` is a Python module for performing accurate and efficient ensemble cross-validation methods from various [projects](https://jaydu1.github.io/overparameterized-ensembling/).


## Features
- The module builds on `scikit-learn`/`sklearn` to provide the most flexibility on various base predictors.
- The module includes functions for creating ensembles of models, training the ensembles using cross-validation, and making predictions with the ensembles. 
- The module also includes utilities for evaluating the performance of the ensembles and the individual models that make up the ensembles.


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn_ensemble_cv import ECV

# Hyperparameters for the base regressor
grid_regr = {    
    'max_depth':np.array([6,7], dtype=int), 
    }
# Hyperparameters for the ensemble
grid_ensemble = {
    'max_features':np.array([0.9,1.]),
    'max_samples':np.array([0.6,0.7]),
    'n_jobs':-1 # use all processors for fitting each ensemble
}

# Build 50 trees and get estimates until 100 trees
res_ecv, info_ecv = ECV(
    X_train, y_train, DecisionTreeRegressor, grid_regr, grid_ensemble, 
    M=50, M_max=100, return_df=True
)
```

It currently supports bagging- and subagging-type ensembles under square loss.
The hyperparameters of the base predictor are listed at [`sklearn.tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) and the hyperparameters of the ensemble are listed at [`sklearn.ensemble.BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html).
Using other sklearn Regressors (`regr.is_regressor = True`) as base predictors is also supported.

## Cross-validation methods

This project is currently in development. More CV methods will be added shortly.

- [x] split CV
- [x] K-fold CV
- [x] ECV
- [x] GCV
- [x] CGCV
- [x] CGCV non-square loss
- [ ] ALOCV

## Usage

The module can be installed via PyPI:
```cmd
pip install sklearn-ensemble-cv
```

The [document](https://sklearn-ensemble-cv.readthedocs.io/en/latest/?badge=latest) is available.
Check out Jupyter Notebook tutorials in the [document](https://sklearn-ensemble-cv.readthedocs.io/en/latest/?badge=latest):

Name | Description
---|---
[basics](https://sklearn-ensemble-cv.readthedocs.io/en/latest/tutorials/basics.html) | Basics about how to apply ECV/CGCV on risk estimation and hyperparameter tuning for ensemble learning.
[cgcv_l1_huber](https://sklearn-ensemble-cv.readthedocs.io/en/latest/tutorials/cgcv_l1_huber.html) | Custom CGCV for M-estimator: l1-regularized Huber ensembles.
[multitask](https://sklearn-ensemble-cv.readthedocs.io/en/latest/tutorials/multitask.html) | Apply ECV on risk estimation and hyperparameter tuning for multi-task ensemble learning.
[random_forests](https://sklearn-ensemble-cv.readthedocs.io/en/latest/tutorials/random_forests.html) | Apply ECV on model selection of random forests via a simple utility function.

The code is tested with `scikit-learn == 1.3.1`.





## Citation

If you find this package useful for your research, please consider citing our research paper: 

Method|Reference
---|---
ECV|Du, J. H., Patil, P., Roeder, K., & Kuchibhotla, A. K. (2024). Extrapolated cross-validation for randomized ensembles. Journal of Computational and Graphical Statistics, 1-12.
GCV|Du, J. H., Patil, P., & Kuchibhotla, A. K. (2023). Subsample ridge ensembles: equivalences and generalized cross-validation. In Proceedings of the 40th International Conference on Machine Learning (pp. 8585-8631).<br>Patil, P., & Du, J. H. (2024). Generalized equivalences between subsampling and ridge regularization. Advances in Neural Information Processing Systems, 36.
CGCV | Bellec, P. C., Du, J. H., Koriyama, T., Patil, P., & Tan, K. (2024). Corrected generalized cross-validation for finite ensembles of penalized estimators. Journal of the Royal Statistical Society Series B: Statistical Methodology, qkae092.
CGCV (non-square loss)|Koriyama, T., Patil, P., Du, J. H., Tan, K., & Bellec, P. C. (2024). Precise asymptotics of bagging regularized M-estimators. arXiv preprint arXiv:2409.15252.
