# Ensemble-cross-validation


`sklearn_ensemble_cv` is a Python module for performing accurate and efficient ensemble cross-validation methods from various [projects](https://jaydu1.github.io/overparameterized-ensembling/).


## Features
- The module builds on `scikit-learn`/`sklearn` to provide most flexibity on various base predictors.
- The module includes functions for creating ensembles of models, training the ensembles using cross-validation, and making predictions with the ensembles. 
- The module also includes utilities for evaluating the performance of the ensembles and the individual models that make up the ensembles.



# Cross-validation methods

This project is currently in development. More CV methods will be added shortly.

- [x] split CV
- [ ] K-fold CV
- [x] ECV
- [ ] GCV
- [ ] CGCV


# Usage

Check out Jupyter notebook [demo.ipynb](https://github.com/jaydu1/ensemble-cross-validation/blob/main/demo.ipynb) about how to apply ECV on risk estimation and hyperparameter tuning for ensemble learning.