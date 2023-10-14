import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn_ensemble_cv.ensemble import Ensemble
from sklearn_ensemble_cv.utils import estimate_null_risk, process_grid
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
n_jobs = 16

import warnings
warnings.filterwarnings('ignore') 


def fit_ensemble(regr=None,kwargs_regr={},kwargs_ensemble={}):
    if regr is None:
        regr = DecisionTreeRegressor
    return Ensemble(estimator=regr(**kwargs_regr), **kwargs_ensemble)


############################################################################
#
# Sample-split and K-fold cross-validation
#
############################################################################

def comp_empirical_val(
        X_train, Y_train, X_val, Y_val, regr, kwargs_regr={}, kwargs_ensemble={}, M=20, M0=20,
        n_jobs=-1, X_test=None, Y_test=None, **kwargs, 
        ):
    '''
    Compute the empirical ECV estimate for a given ensemble model.

    Parameters
    ----------
    X_train,Y_train : numpy.array
        The training samples.
    X_val,Y_val : numpy.array
        The validation samples.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The maximum ensemble size to consider.
    M0 : int, optional
        The number of estimators to use for the ECV estimate.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test,Y_test : numpy.array, optional
        The test samples.
    kwargs : dict, optional
        Additional keyword arguments for the risk estimate.
    
    Return:
    ----------
    risk_ecv : numpy.array
        The empirical ECV estimate.
    '''
    
    kwargs_ensemble = {**{'random_state':0}, **kwargs_ensemble}
    kwargs_est = {**{'re_method':'AVG', 'eta':None}, **kwargs}
    if M0>M:
        raise ValueError('M0 must be less than or equal to M.')
    kwargs_ensemble['n_estimators'] = M

    regr = fit_ensemble(regr,kwargs_regr,kwargs_ensemble).fit(X_train, Y_train)
    risk_val = regr.compute_risk(X_val, Y_val, M_test=None, return_df=False, n_jobs=n_jobs, **kwargs_est)

    if X_val is not None and Y_test is not None:
        risk_test = regr.compute_risk(X_test, Y_test, M, n_jobs=n_jobs, **kwargs_est)
        return regr, (risk_val, risk_test)
    else:
        return regr, risk_val


def splitCV(
        X_train, Y_train, regr, grid_regr={}, grid_ensemble={}, kwargs_regr={}, kwargs_ensemble={},
        M=20, return_df=False, n_jobs=-1, X_test=None, Y_test=None, kwargs_est={}, **kwargs
        ):
    '''
    Sample-split cross-validation for ensemble models.

    Parameters
    ----------
    X_train,Y_train : numpy.array
        The training samples.
    regr : object
        The base estimator to use for the ensemble model.        
    grid_regr : pandas.DataFrame
        The grid of hyperparameters for the base estimator.
    grid_ensemble : pandas.DataFrame
        The grid of hyperparameters for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The ensemble size to build.
    return_df : bool, optional
        If True, returns the results as a pandas.DataFrame object.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test,Y_test : numpy.array, optional
        The test samples. It may be useful to be used for comparing the
        performance of different cross-validation methods.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    kwargs : dict, optional
        Additional keyword arguments for `train_test_split`; see 
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        for more details.
    '''
    if not grid_regr and not grid_ensemble:
        raise ValueError('grid_regr and grid_ensemble cannot both be empty.')
    
    if type(grid_regr) is not type(grid_ensemble):
        raise ValueError('grid_regr and grid_ensemble must be of the same type.')
    
    if isinstance(grid_regr, dict):
        grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble = process_grid(
            grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble)
            
    test = X_test is not None and Y_test is not None
    n_res = 2*M if test else M
    n_grid = len(grid_regr)
    res_risk = np.full((n_grid,n_res), np.inf)

    rs = ShuffleSplit(1, **kwargs)
    id_train, id_val = next(rs.split(X_train, Y_train))
    _X_train, _X_val, _Y_train, _Y_val = X_train[id_train], X_train[id_val], Y_train[id_train], Y_train[id_val]
    
    for i in range(n_grid):
        params_ensemble = grid_ensemble[i]
        params_regr = grid_regr[i]
        
        _, res = comp_empirical_val(
            _X_train, _Y_train, _X_val, _Y_val, regr, kwargs_regr, kwargs_ensemble, M,
            n_jobs, X_test, Y_test, **kwargs_est
        )
        res_risk[i, :] = np.r_[res]

    if return_df:
        cols = np.char.add(['risk_val-']*M, np.char.mod('%d', 1+np.arange(M)))
        if test:
            cols = np.append(cols, np.char.add(['risk_test-']*M, np.char.mod('%d', 1+np.arange(M))))
        res_splitcv = pd.concat([pd.DataFrame(grid_regr), pd.DataFrame(grid_ensemble),
                             pd.DataFrame(res_risk, columns=cols)
                             ] ,axis=1)
    else:
        if test:            
            res_splitcv = (res_risk[:,:M], res_risk[:,M:])
        else:
            res_splitcv = res_risk

    j = np.nanargmin(2 * res_risk[:,1] - res_risk[:,0])

    info = {
        'best_params_regr': {**params_regr, **grid_regr[j]},
        'best_params_ensemble': {**params_ensemble, **grid_ensemble[j]},
        'split_params':{
            'index_train':id_train, 
            'index_val':id_val,
            'test_size':rs.test_size,
            'random_state':rs.random_state
        }
    }
    return res_splitcv, info
    


    
    



############################################################################
#
# Out-of-bag cross-validation
#
############################################################################


def comp_empirical_ecv(
        X_train, Y_train, regr, kwargs_regr={}, kwargs_ensemble={}, M=20, M0=20,
        n_jobs=-1, X_test=None, Y_test=None, **kwargs, 
        ):
    '''
    Compute the empirical ECV estimate for a given ensemble model.

    Parameters
    ----------
    X_train,Y_train : numpy.array
        The training samples.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The maximum ensemble size to consider.
    M0 : int, optional
        The number of estimators to use for the ECV estimate.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test,Y_test : numpy.array, optional
        The test samples.
    kwargs : dict, optional
        Additional keyword arguments for the risk estimate.
    
    Return:
    ----------
    risk_ecv : numpy.array
        The empirical ECV estimate.
    '''
    
    kwargs_ensemble = {**{'random_state':0}, **kwargs_ensemble}
    kwargs_est = {**{'re_method':'AVG', 'eta':None}, **kwargs}
    if M0>M:
        raise ValueError('M0 must be less than or equal to M.')
    kwargs_ensemble['n_estimators'] = M
    # # null predictor
    # if np.isinf(phi_s):
    #     risk_ecv = np.full(M if np.isscalar(M) else len(M), estimate_null_risk(Y))
    # else:
    regr = fit_ensemble(regr,kwargs_regr,kwargs_ensemble).fit(X_train, Y_train)
    risk_ecv = regr.compute_ecv_estimate(X_train, Y_train, M, M0=M0, n_jobs=n_jobs, **kwargs_est)


    if X_test is not None and Y_test is not None:
        risk_val = regr.compute_risk(X_test, Y_test, M, n_jobs=n_jobs, **kwargs_est)
        return regr, (risk_ecv, risk_val)
    else:
        return regr, risk_ecv

    



def ECV(
        X_train, Y_train, regr, grid_regr={}, grid_ensemble={}, kwargs_regr={}, kwargs_ensemble={},
        M=20, M0=20, M_max=np.inf, delta=0., return_df=False, n_jobs=-1, X_test=None, Y_test=None, 
        kwargs_est={}, **kwargs
        ):
    '''
    Cross-validation for ensemble models using the empirical ECV estimate.

    Parameters
    ----------
    X_train,Y_train : numpy.array
        The training samples.
    grid : pandas.DataFrame
        The grid of hyperparameters to search over.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M0 : int, optional
        The number of estimators to use for the ECV estimate.
    M_max : int, optional
        The maximum ensemble size to consider for the tuned ensemble.
    delta : float, optional
        The suboptimality parameter for the ensemble size tuning by ECV.
    return_df : bool, optional
        If True, returns the results as a pandas.DataFrame object.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test,Y_val : numpy.array, optional
        The validation samples. It may be useful to be used for comparing the 
        performance of ECV with other cross-validation methods that requires sample-splitting.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    '''
    if not grid_regr and not grid_ensemble:
        raise ValueError('grid_regr and grid_ensemble cannot both be empty.')
    
    if type(grid_regr) is not type(grid_ensemble):
        raise ValueError('grid_regr and grid_ensemble must be of the same type.')
    
    if isinstance(grid_regr, dict):
        grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble = process_grid(
            grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble)

    test = X_test is not None and Y_test is not None
    n_res = 2*M if test else M
    n_grid = len(grid_regr)
    res_risk = np.full((n_grid,n_res), np.inf)
    
    for i in range(n_grid):
        params_ensemble = grid_ensemble[i]
        params_regr = grid_regr[i]

        _, res = comp_empirical_ecv(
            X_train, Y_train, regr, 
            {**kwargs_regr, **params_regr}, {**kwargs_ensemble, **params_ensemble},
            M, M0, n_jobs, X_test, Y_test, **kwargs_est
        )
        res_risk[i, :] = np.r_[res]

    if return_df:
        cols = np.char.add(['risk_val-']*M, np.char.mod('%d', 1+np.arange(M)))
        if test:
            cols = np.append(cols, np.char.add(['risk_test-']*M, np.char.mod('%d', 1+np.arange(M))))
        res_ecv = pd.concat([pd.DataFrame(grid_regr), pd.DataFrame(grid_ensemble),
                             pd.DataFrame(res_risk, columns=cols)
                             ] ,axis=1)
    else:
        if test:            
            res_ecv = (res_risk[:,:M], res_risk[:,M:])
        else:
            res_ecv = res_risk

    j = np.nanargmin(2 * res_risk[:,1] - res_risk[:,0])

    if delta==0.:
        M_hat = np.inf
    else:
        M_hat = int(np.ceil(2 / delta * (res_risk[j,0] - res_risk[j,1])))
    best_n_estimators_ = np.minimum(M_hat, M_max)

    info = {
        'delta': delta,
        'best_params_regr': {**params_regr, **grid_regr[j]},
        'best_params_ensemble': {**params_ensemble, **grid_ensemble[j]},
        'best_n_estimators': best_n_estimators_,
        'M_max':M_max
    }
    return res_ecv, info
    
    





# def compute_prediction_risk(X, Y, X_test, Y_test, method, param, 
#                             M, k_list=None, nu=0.5, bootstrap=False, **kwargs):
#     n, p = X.shape
#     n_base = int(n**nu)

#     if k_list is not None:
#         k_list = np.array(k_list)
#     else:
#         k_list = np.arange(n_base, n+1, n_base)
#         if n!=k_list[-1]:
#             k_list = np.append(k_list, n)
#     if 0 not in k_list:
#         k_list = np.insert(k_list,0,0)
    
#     res_val = np.full((len(k_list),M), np.inf)
#     res_test = np.full((len(k_list),M), np.inf)
    
#     for j,k in enumerate(k_list):
#         res_val[j,:], res_test[j,:] = comp_empirical_oobcv(X, Y, X_test, Y_test, 
#             p/k, method, param, M, M0=1, M_test=M, bootstrap=bootstrap, **kwargs)

#     return k_list, res_val, res_test    