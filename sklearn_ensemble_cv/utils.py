import numpy as np
import pandas as pd
from itertools import product



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
    sq_err : 1d-array
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


def estimate_null_risk(Y):
    '''
    Estimate the null risk of the data for regression problems.
    '''
    mu = 0.
    return np.mean((Y-mu)**2)




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
