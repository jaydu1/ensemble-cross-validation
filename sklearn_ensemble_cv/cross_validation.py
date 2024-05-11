import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn_ensemble_cv.ensemble import Ensemble
from sklearn_ensemble_cv.utils import check_input, process_grid
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
        X_train, Y_train, X_val, Y_val, regr, kwargs_regr={}, kwargs_ensemble={}, M=20,
        n_jobs=-1, X_test=None, Y_test=None, _check_input=True, **kwargs_est, 
        ):
    '''
    Compute the empirical ECV estimate for a given ensemble model.

    Parameters
    ----------
    X_train, Y_train : numpy.array
        The training samples.
    X_val, Y_val : numpy.array
        The validation samples.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The maximum ensemble size to consider.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test, Y_test : numpy.array, optional
        The test samples.
    _check_input : bool, optional
        If True, check the input arguments.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    
    Returns
    ----------
    risk_ecv : numpy.array
        The empirical ECV estimate.
    '''
    if _check_input:
        kwargs_regr, kwargs_ensemble, kwargs_est = check_input(kwargs_regr, kwargs_ensemble, kwargs_est, M)

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
    X_train, Y_train : numpy.array
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
    X_test, Y_test : numpy.array, optional
        The test samples. It may be useful to be used for comparing the
        performance of different cross-validation methods.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    kwargs : dict, optional
        Additional keyword arguments for `ShuffleSplit`; see 
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
        for more details.
    '''
    grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est = process_grid(
        grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est, M)
            
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
            _X_train, _Y_train, _X_val, _Y_val, regr, 
            {**kwargs_regr, **params_regr}, {**kwargs_ensemble, **params_ensemble},
            M, n_jobs, X_test, Y_test,  _check_input=False, **kwargs_est
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

    j, M_best = np.unravel_index(np.nanargmin(res_risk[:,:M]), res_risk[:,:M].shape)
    M_best += 1

    info = {
        'best_params_regr': {**kwargs_regr, **grid_regr[j]},
        'best_params_ensemble': {**kwargs_ensemble, **grid_ensemble[j]},
        'best_n_estimators': M_best,
        'best_params_index':j,
        'best_score':res_risk[j, M_best-1],
        'split_params':{
            'index_train':id_train, 
            'index_val':id_val,
            'test_size':rs.test_size,
            'random_state':rs.random_state
        }
    }
    return res_splitcv, info
    


def KFoldCV(
        X_train, Y_train, regr, grid_regr={}, grid_ensemble={}, kwargs_regr={}, kwargs_ensemble={},
        M=20, return_df=False, n_jobs=-1, X_test=None, Y_test=None, kwargs_est={}, **kwargs
        ):
    '''
    Sample-split cross-validation for ensemble models.

    Parameters
    ----------
    X_train, Y_train : numpy.array
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
    X_test, Y_test : numpy.array, optional
        The test samples. It may be useful to be used for comparing the
        performance of different cross-validation methods.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    kwargs : dict, optional
        Additional keyword arguments for `KFold`; see 
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        for more details.
    '''
    grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est = process_grid(
        grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est, M)
            
    test = X_test is not None and Y_test is not None
    n_res = 2*M if test else M
    n_grid = len(grid_regr)
    
    kf = KFold(**kwargs)
    n_splits = kf.get_n_splits(X_train)
    res_risk_all = np.full((n_grid,n_res,n_splits), np.inf)

    for fold, (id_train, id_val) in enumerate(kf.split(X_train)):

        _X_train, _X_val, _Y_train, _Y_val = X_train[id_train], X_train[id_val], Y_train[id_train], Y_train[id_val]
        
        for i in range(n_grid):
            params_ensemble = grid_ensemble[i]
            params_regr = grid_regr[i]
            
            _, res = comp_empirical_val(
                _X_train, _Y_train, _X_val, _Y_val, regr, 
                {**kwargs_regr, **params_regr}, {**kwargs_ensemble, **params_ensemble}, 
                M, n_jobs, X_test, Y_test,  _check_input=False, **kwargs_est
            )
            res_risk_all[i, :, fold] = np.r_[res]

    res_risk = np.mean(res_risk_all, axis=2)

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

    j, M_best = np.unravel_index(np.nanargmin(res_risk[:,:M]), res_risk[:,:M].shape)
    M_best += 1

    info = {
        'best_params_regr': {**kwargs_regr, **grid_regr[j]},
        'best_params_ensemble': {**kwargs_ensemble, **grid_ensemble[j]},
        'best_n_estimators': M_best,
        'best_params_index':j,
        'best_score':res_risk[j, M_best-1],

        'val_score':res_risk_all[:,:M],
        'test_score':None if not test else res_risk_all[:,M:],
        'split_params':{
            'n_splits':n_splits,
            'random_state':kf.random_state,
            'shuffle':kf.shuffle,            
        }
    }
    return res_splitcv, info  



############################################################################
#
# Out-of-bag cross-validation
#
############################################################################


def comp_empirical_ecv(
        X_train, Y_train, regr, kwargs_regr={}, kwargs_ensemble={}, M=20, M0=20, M_max=np.inf,
        n_jobs=-1, X_test=None, Y_test=None, _check_input=True, **kwargs_est, 
        ):
    '''
    Compute the empirical ECV estimate for a given ensemble model.

    Parameters
    ----------
    X_train, Y_train : numpy.array
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
    M_max : int, optional
        The maximum ensemble size to consider for the tuned ensemble.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test, Y_test : numpy.array, optional
        The test samples.
    _check_input : bool, optional
        If True, check the input arguments.        
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    
    Returns
    ----------
    risk_ecv : numpy.array
        The empirical ECV estimate.
    '''
    if _check_input:
        if M0>M:
            raise ValueError('M0 must be less than or equal to M.')
        if np.isinf(M_max):
            M_max = np.append(np.arange(M)+1, np.inf)
        elif np.isscalar(M_max):
            M_max = np.arange(M_max)+1

        kwargs_regr, kwargs_ensemble, kwargs_est = check_input(kwargs_regr, kwargs_ensemble, kwargs_est, M)
    regr = fit_ensemble(regr,kwargs_regr,kwargs_ensemble).fit(X_train, Y_train)
    risk_ecv = regr.compute_ecv_estimate(X_train, Y_train, M_max, M0=M0, n_jobs=n_jobs, **kwargs_est)

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
    X_train, Y_train : numpy.array
        The training samples.
    grid : pandas.DataFrame
        The grid of hyperparameters to search over.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The ensemble size to build.
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
    X_test, Y_test : numpy.array, optional
        The validation samples. It may be useful to be used for comparing the 
        performance of ECV with other cross-validation methods that requires sample-splitting.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    '''
    grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est = process_grid(
        grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est, M)

    if M0>M:
        raise ValueError('M0 must be less than or equal to M.')
    if np.isinf(M_max):
        M_max = np.append(np.arange(M)+1, np.inf)
    elif np.isscalar(M_max):
        M_max = np.arange(M_max)+1
    n_M_max = len(M_max)

    test = X_test is not None and Y_test is not None
    n_res = n_M_max+M if test else n_M_max
    n_grid = len(grid_regr)
    res_risk = np.full((n_grid, n_res), np.inf)

    for i in range(n_grid):
        params_ensemble = grid_ensemble[i]
        params_regr = grid_regr[i]

        _, res = comp_empirical_ecv(
            X_train, Y_train, regr, 
            {**kwargs_regr, **params_regr}, {**kwargs_ensemble, **params_ensemble},
            M, M0, M_max, n_jobs, X_test, Y_test, _check_input=False, **kwargs_est
        )
        res_risk[i, :] = np.r_[res]

    if return_df:
        cols = np.char.add(['risk_val-']*n_M_max, np.char.mod('%d', 1+np.arange(n_M_max)))
        if np.isinf(M_max[-1]):
            cols[-1] = 'risk_val-inf'
        if test:
            cols = np.append(cols, np.char.add(['risk_test-']*M, np.char.mod('%d', 1+np.arange(M))))
        res_ecv = pd.concat([pd.DataFrame(grid_regr), pd.DataFrame(grid_ensemble),
                             pd.DataFrame(res_risk, columns=cols)
                             ] ,axis=1)
    else:
        if test:            
            res_ecv = (res_risk[:,:n_M_max], res_risk[:,n_M_max:])
        else:
            res_ecv = res_risk

    j, M_best = np.unravel_index(np.nanargmin(res_risk[:,:M]), res_risk[:,:M].shape)
    M_best += 1

    if delta==0.:
        M_hat = np.inf
    else:
        M_hat = int(np.ceil(2 / (delta + 2/M_max[-1]*(res_risk[j,0] - res_risk[j,1])) * 
                            (res_risk[j,0] - res_risk[j,1])))
    M_best_ext = np.minimum(M_hat, M_max[-1])
    if not np.isinf(M_best_ext):
        M_best_ext = int(M_best_ext)
    
    info = {        
        'best_params_regr': {**kwargs_regr, **grid_regr[j]},
        'best_params_ensemble': {**kwargs_ensemble, **grid_ensemble[j]},
        'best_n_estimators': M_best,
        'best_params_index':j,
        'best_score':res_risk[j, M_best-1],

        'delta': delta,
        'M_max':M_max[-1],
        'best_n_estimators_extrapolate': M_best_ext,
        'best_score_extrapolate': res_risk[j,n_M_max-1] if np.isinf(M_best_ext) else res_risk[j, M_best_ext-1],        
    }
    return res_ecv, info
    
    

############################################################################
#
# Generalized cross-validation
#
############################################################################


def comp_empirical_gcv(
        X_train, Y_train, regr, kwargs_regr={}, kwargs_ensemble={}, M=20, M0=20, M_max=np.inf, 
        corrected=True, type='full',
        n_jobs=-1, X_test=None, Y_test=None, _check_input=True, **kwargs_est, 
        ):
    '''
    Compute the empirical GCV or CGCV estimate for a given ensemble model.

    Parameters
    ----------
    X_train, Y_train : numpy.array
        The training samples.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The maximum ensemble size to consider.
    corrected : bool, optional
        If True, compute the corrected GCV estimate.
    type : str, optional
        The type of GCV or GCV estimate to compute. It can be either 'full' or 'union' for naive GCV,
        and 'full' or 'ovlp' for CGCV.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test, Y_test : numpy.array, optional
        The test samples.
    _check_input : bool, optional
        If True, check the input arguments.        
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    
    Returns
    ----------
    risk_ecv : numpy.array
        The empirical ECV estimate.
    '''
    if _check_input:
        if M0>M:
            raise ValueError('M0 must be less than or equal to M.')
        if np.isinf(M_max):
            M_max = np.append(np.arange(M)+1, np.inf)
        elif np.isscalar(M_max):
            M_max = np.arange(M_max)+1
    
        kwargs_regr, kwargs_ensemble, kwargs_est = check_input(kwargs_regr, kwargs_ensemble, kwargs_est, M)

    regr = fit_ensemble(regr,kwargs_regr,kwargs_ensemble).fit(X_train, Y_train)
    if corrected:
        risk_gcv = regr.compute_cgcv_estimate(X_train, Y_train, M0, type, n_jobs=n_jobs, **kwargs_est)
    else:
        risk_gcv = regr.compute_gcv_estimate(X_train, Y_train, M0, type, n_jobs=n_jobs, **kwargs_est)

    risk_gcv = regr.extrapolate(risk_gcv, M_max)
    
    if X_test is not None and Y_test is not None:        
        risk_val = regr.compute_risk(X_test, Y_test, M, n_jobs=n_jobs, **kwargs_est)
        return regr, (risk_gcv, risk_val)
    else:
        return regr, risk_gcv
    

def GCV(
        X_train, Y_train, regr, grid_regr={}, grid_ensemble={}, kwargs_regr={}, kwargs_ensemble={},
        M=20, M0=20, M_max=np.inf, corrected=True, type='full', return_df=False, n_jobs=-1, X_test=None, Y_test=None, 
        kwargs_est={}, **kwargs
        ):
    '''
    Cross-validation for ensemble models using the empirical ECV estimate.
    Currently, only the GCV estimates for the Ridge, Lasso, and ElasticNet are implemented.

    Parameters
    ----------
    X_train, Y_train : numpy.array
        The training samples.
    grid : pandas.DataFrame
        The grid of hyperparameters to search over.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.
    M : int, optional
        The ensemble size to build.
    corrected : bool, optional
        If True, compute the corrected GCV estimate.
    type : str, optional
        The type of GCV or GCV estimate to compute. It can be either 'full' or 'union' for naive GCV,
        and 'full' or 'ovlp' for CGCV.
    return_df : bool, optional
        If True, returns the results as a pandas.DataFrame object.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_test, Y_test : numpy.array, optional
        The validation samples. It may be useful to be used for comparing the 
        performance of ECV with other cross-validation methods that requires sample-splitting.
    kwargs_est : dict, optional
        Additional keyword arguments for the risk estimate.
    '''
    grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est = process_grid(
        grid_regr, kwargs_regr, grid_ensemble, kwargs_ensemble, kwargs_est, M)

    if M0>M:
        raise ValueError('M0 must be less than or equal to M.')
    if np.isinf(M_max):
        M_max = np.append(np.arange(M)+1, np.inf)
    elif np.isscalar(M_max):
        M_max = np.arange(M_max)+1
    n_M_max = len(M_max)

    test = X_test is not None and Y_test is not None
    n_res = n_M_max+M if test else n_M_max
    n_grid = len(grid_regr)
    res_risk = np.full((n_grid, n_res), np.inf)

    for i in range(n_grid):
        params_ensemble = grid_ensemble[i]
        params_regr = grid_regr[i]
        
        _, res = comp_empirical_gcv(
            X_train, Y_train, regr, 
            {**kwargs_regr, **params_regr}, {**kwargs_ensemble, **params_ensemble},
            M, M0, M_max, corrected, type, n_jobs, X_test, Y_test,  _check_input=False, **kwargs_est
        )
        res_risk[i, :] = np.r_[res]

    if return_df:
        cols = np.char.add(['risk_val-']*n_M_max, np.char.mod('%d', 1+np.arange(n_M_max)))
        if np.isinf(M_max[-1]):
            cols[-1] = 'risk_val-inf'
        if test:
            cols = np.append(cols, np.char.add(['risk_test-']*M, np.char.mod('%d', 1+np.arange(M))))        
        res_gcv = pd.concat([pd.DataFrame(grid_regr), pd.DataFrame(grid_ensemble),
                             pd.DataFrame(res_risk, columns=cols)
                             ] ,axis=1)
    else:
        if test:            
            res_gcv = (res_risk[:,:M], res_risk[:,M:])
        else:
            res_gcv = res_risk

    j, M_best = np.unravel_index(np.nanargmin(res_risk[:,:M]), res_risk[:,:M].shape)
    M_best += 1

    info = {
        'best_params_regr': {**kwargs_regr, **grid_regr[j]},
        'best_params_ensemble': {**kwargs_ensemble, **grid_ensemble[j]},
        'best_n_estimators': M_best,
        'best_params_index':j,
        'best_score':res_risk[j, M_best-1],
    }

    return res_gcv, info




