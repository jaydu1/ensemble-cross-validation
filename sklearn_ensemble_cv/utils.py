import os
import random
import numpy as np
import pandas as pd
from itertools import product, combinations


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)



def median_of_means(x, eta=None):
    '''
    Compute the median of means of the given data.

    Parameters
    ----------
    x : 1d-array
        The data.
    eta : float
        The parameter for the median of means. If None, it is set to 1/n.

    Returns
    -------
    mom : float
        The median of means.
    '''
    n = len(x)
    if eta is None:
        eta = 1/n
    B = int(np.maximum(
            np.minimum(np.ceil(8 * np.log(1/eta)), n), 1))
    ids = np.random.permutation(np.arange(n))
    ids_list = np.array_split(ids, B)
    mom = np.median([np.mean(x[ids]) for ids in ids_list])
    return mom


def risk_estimate(sq_err, axis=None, method='AVG', **kwargs):
    '''
    Compute the risk estimate from the squared error.

    Parameters
    ----------
    sq_err : 2d-array
        The squared error.
    method : str
        The method to use for risk estimation. Either 'AVG' or 'MOM'.
    kwargs : dict
        Additional keyword arguments for the risk estimation method.
    '''

    if len(sq_err)<1:
        return np.nan
    
    if method=='AVG':
        risk = np.mean(sq_err, axis=axis)        
    else:
        risk = np.apply_along_axis(median_of_means, axis, sq_err, **kwargs)
    return risk


def degree_of_freedom(regr, X):
    '''
    Compute the degree of freedom of a fitted regressor.
    
    Parameters
    ----------
    regr : sklearn regressor
        The fitted regressor. Can be Ridge, Lasso, or ElasticNet.
    X : 2d-array
        The input data.

    Returns
    -------
    dof : float
        The degree of freedom.
    '''
    k = X.shape[0]
    if regr.fit_intercept:
        X = np.c_[np.ones((k,1)), X]
        nz_coef = np.r_[regr.intercept_!=0, regr.coef_!=0]
    else:
        nz_coef = regr.coef_!=0
    
    lam = regr.alpha
    method = regr.__class__.__name__

    if method == 'Ridge':
        svds = np.linalg.svd(X, compute_uv=False)
        evds = svds[:k]**2
        dof = np.sum(evds/(evds + lam))

    elif method == 'Lasso':
        dof = np.sum(nz_coef)
            
    elif method=='ElasticNet':
        l1_ratio = regr.l1_ratio
        lam_2 = lam * (1-l1_ratio)

        if np.any(nz_coef):
            svds = np.linalg.svd(X[:,nz_coef], compute_uv=False)
        else:
            svds = np.array([0.])
        evds = svds[:k]**2
        dof = np.sum(evds/(evds + k * lam_2))

    return dof


def estimate_null_risk(Y):
    '''
    Estimate the null risk of the data for regression problems.
    '''
    mu = 0.
    return np.mean((Y-mu)**2)



def _avg_sq_err_M(x, M, M_max, axis=0, **kwargs_est):
    '''
    Compute the average of all combinations of M of all columns.

    Parameters
    ----------
    x : 2d-array
        The data of shape [n,M].
    M : int
        The number of columns to combine.
    M_max : int
        The maximum combinations number of columns to combine.

    Returns
    -------
    avg_sq_err : float
        The average squared error of the M-ensemble.
    '''
    if M==1:
        return np.mean(x**2, axis=1)
    else:
        iter = 0
        err = []
        for id in combinations(np.arange(x.shape[1]), M):
            if iter >= M_max:
                break
            err.append(np.mean(x[:,id], axis=1))
            iter += 1
        err = np.c_[err]**2
        return np.mean(err, axis=0)


def avg_sq_err(err, M_max=None):
    '''
    Compute the average squared error.

    Parameters
    ----------
    err : 2d-array
        The squared errors of shape [n,M].

    Returns
    -------
    risk : 1d-array
        The estimated squared errors of the M-ensembles.
    '''
    if M_max is None:
        M_max = np.ones(err.shape[1]) * 500
        M_max[np.arange(err.shape[1])>10] = 10
        M_max = M_max.astype(int)
    risk = np.fromiter((_avg_sq_err_M(err, M+1, M_max[M]) for M in np.arange(err.shape[1])), 
                       dtype=np.dtype((float, err.shape[0]))).T
    return risk

####################################################################################################
#
# Grid processing functions
#
####################################################################################################

def split_grid(raw_grid, raw_kwarg):
    '''
    Split the grid and kwarg into two dictionaries.

    Parameters
    ----------
    raw_grid : dict
        A dictionary of lists of parameters, possibly with fixed parameters.
    raw_kwarg : dict
        A dictionary of fixed parameters.
    
    Returns
    -------
    grid : dict
        A dictionary of lists of parameters to tune.
    kwarg : dict
        A dictionary of fixed parameters.
    '''

    grid = {i:j for i,j in raw_grid.items() if not np.isscalar(j)}
    kwarg = {i:j for i,j in raw_grid.items() if np.isscalar(j) or len(j)==1}

    if raw_kwarg.keys() & kwarg.keys():
        raise ValueError('Grid and kwarg cannot have common keys.')
    kwarg = {**kwarg, **raw_kwarg}
    return grid, kwarg


def make_grid(dict_regr, dict_ensemble=None):
    '''
    Create a dataframe with all combinations of parameters in dict_params.

    Parameters
    ----------
    dict_regr : dict
        A dictionary of parameter names and their possible values for the base regressor.
    dict_ensemble : dict
        A dictionary of parameter names and their possible values for the ensemble model.

    Returns
    -------
    config_list_regr : list
        A list of dictionaries, where each dictionary represents one configuration for the base regressor.
    config_list_ensemble : list
        A list of dictionaries, where each dictionary represents one configuration for the ensemble model.
    '''    
    # Get all combinations of parameter values
    param_values = list(product(*(list(dict_regr.values())+list(dict_ensemble.values()))))

    # Create a list of dictionaries, where each dictionary represents one configuration
    config_list_regr = [dict(zip(dict_regr.keys(), values[:len(dict_regr)])) for values in param_values]
    if dict_ensemble is not None:
        config_list_ensemble = [dict(zip(dict_ensemble.keys(), values[len(dict_regr):])) for values in param_values]
        return config_list_regr, config_list_ensemble
    else:
        return config_list_regr


def process_grid(grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est, M):
    '''
    Process the grid and kwarg into two dictionaries.

    Parameters
    ----------
    grid_regr : dict
        A dictionary of lists of parameters for the base regressor, possibly with fixed parameters.
    kwargs_regr : dict
        A dictionary of fixed parameters for the base regressor.
    grid_ensemble : dict
        A dictionary of lists of parameters for the ensemble model, possibly with fixed parameters.
    kwargs_ensemble : dict
        A dictionary of fixed parameters for the ensemble model.
    kwargs_est : dict
        Additional keyword arguments for the risk estimate.
    M : int
        The ensemble size.

    Returns
    -------
    grid_regr : dict
        A dictionary of lists of parameters to tune for the base regressor.
    kwargs_regr : dict
        A dictionary of fixed parameters for the base regressor.
    grid_ensemble : dict
        A dictionary of lists of parameters to tune for the ensemble model.
    kwargs_ensemble : dict
        A dictionary of fixed parameters for the ensemble model.
    kwargs_est : dict
        Additional keyword arguments for the risk estimate.    
    '''
    if not grid_regr and not grid_ensemble:
        raise ValueError('grid_regr and grid_ensemble cannot both be empty.')
    
    if type(grid_regr) is not type(grid_ensemble):
        raise ValueError('grid_regr and grid_ensemble must be of the same type.')
    
    if isinstance(grid_regr, dict):
        grid_regr, kwargs_regr = split_grid(grid_regr, kwargs_regr)
        grid_ensemble, kwargs_ensemble = split_grid(grid_ensemble, kwargs_ensemble)
        grid_regr, grid_ensemble = make_grid(grid_regr, grid_ensemble)

    kwargs_ensemble = {**{'random_state':0}, **kwargs_ensemble}
    kwargs_regr, kwargs_ensemble, kwargs_est = check_input(kwargs_regr, kwargs_ensemble, kwargs_est, M)
    return grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est



def check_input(kwargs_regr, kwargs_ensemble, kwargs_est, M):
    '''
    Check the input parameters for the risk estimate.

    Parameters
    ----------
    kwargs_regr : dict
        A dictionary of fixed parameters for the base regressor.
    kwargs_ensemble : dict
        A dictionary of fixed parameters for the ensemble model.
    kwargs_est : dict
        Additional keyword arguments for the risk estimate.
    M : int
        The ensemble size.

    Returns
    -------
    kwargs_regr : dict
        The updated fixed parameters for the base regressor.
    kwargs_ensemble : dict
        The updated fixed parameters for the ensemble model.
    kwargs_est : dict
        The updated additional keyword arguments for the risk estimate.
    '''
    kwargs_est = {**{'re_method':'AVG', 'eta':None}, **kwargs_est}
    kwargs_ensemble['n_estimators'] = M

    return kwargs_regr, kwargs_ensemble, kwargs_est
