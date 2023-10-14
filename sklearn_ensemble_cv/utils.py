import numpy as np
import pandas as pd
from itertools import product



def risk_estimate(sq_err, method='AVG', eta=None, **kwargs):
    '''
    Compute the risk estimate from the squared error.

    Parameters
    ----------
    sq_err : 1d-array
        The squared error.
    method : str
        The method to use for risk estimation. Either 'AVG' or 'MOM'.
    eta : float
        The parameter for 'MOM' estimation.
    '''

    if len(sq_err)<1:
        return np.nan
    
    if method=='AVG':
        risk = np.mean(sq_err)        
    else:
        n = sq_err.shape[0]
        if eta is None:
            eta = 1/n
        B = int(np.maximum(
            np.minimum(np.ceil(8 * np.log(1/eta)), n), 1))
        ids_list = np.array_split(np.random.permutation(np.arange(n)), B)
        risk = np.median([np.mean(sq_err[ids]) for ids in ids_list])
    return risk


def estimate_null_risk(Y):
    '''
    Estimate the null risk of the data for regression problems.
    '''
    mu = 0.
    return np.mean((Y-mu)**2)


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


def make_grid(dict_regr, dict_ensemble):
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
    config_list_ensemble = [dict(zip(dict_ensemble.keys(), values[len(dict_regr):])) for values in param_values]

    return config_list_regr, config_list_ensemble