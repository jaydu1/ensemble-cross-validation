import numpy as np


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


import pandas as pd
from itertools import product


def make_grid(dict_params):
    '''
    Create a dataframe with all combinations of parameters in dict_params.

    Parameters
    ----------
    dict_params : dict
        A dictionary of parameter names and their possible values.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe with all combinations of parameters.
    '''

    dtypes = {k: type(v[0]) for k, v in dict_params.items()}

    # Get all combinations of parameter values
    param_values = list(product(*dict_params.values()))

    # Create a list of dictionaries, where each dictionary represents one configuration
    config_list = [dict(zip(dict_params.keys(), values)) for values in param_values]

    # Convert the list of dictionaries to a dataframe    
    df = pd.DataFrame(config_list).astype(dtypes)

    return df

