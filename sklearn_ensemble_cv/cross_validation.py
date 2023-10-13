import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn_ensemble_cv.ensemble import Ensemble
from sklearn_ensemble_cv.utils import estimate_null_risk
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
n_jobs = 16

import warnings
warnings.filterwarnings('ignore') 

    


############################################################################
#
# Out-of-bag cross-validation
#
############################################################################


def comp_empirical_ecv(
        X_train, Y_train, 
        regr, kwargs_regr={}, 
        M=20, M0=20, kwargs_ensemble={},
        n_jobs=-1, X_val=None, Y_val=None, **kwargs, 
        ):
    '''
    Compute the empirical ECV estimate for a given ensemble model.

    Parameters
    ----------
    X_train,Y_train : numpy.array
        The traning samples.
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
    kwargs_est : dict, optional
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
    if regr is None:
        regr = DecisionTreeRegressor
    # # null predictor
    # if np.isinf(phi_s):
    #     risk_ecv = np.full(M if np.isscalar(M) else len(M), estimate_null_risk(Y))
    # else:
    regr = Ensemble(estimator=regr(**kwargs_regr), n_estimators=M, **kwargs_ensemble).fit(X_train, Y_train)
    risk_ecv = regr.compute_ecv_estimate(X_train, Y_train, M, M0=M0, n_jobs=n_jobs, **kwargs_est)


    if X_val is not None and Y_val is not None:
        risk_val = regr.compute_risk(X_val, Y_val, M, n_jobs=n_jobs, **kwargs_est)
        return regr, (risk_ecv, risk_val)
    else:
        return regr, risk_ecv

    



def cross_validation_ecv(
        X_train, Y_train, grid,
        regr, kwargs_regr={},
        M=20, M0=20,  kwargs_ensemble={}, 
        M_max=np.inf, delta=0., return_df=False, n_jobs=-1, X_val=None, Y_val=None, **kwargs
        ):
    '''
    Cross-validation for ensemble models using the empirical ECV estimate.

    Parameters
    ----------
    X_train,Y_train : numpy.array
        The traning samples.
    grid : pandas.DataFrame
        The grid of hyperparameters to search over.
    regr : object
        The base estimator to use for the ensemble model.
    kwargs_regr : dict, optional
        Additional keyword arguments for the base estimator.
    M : int, optional
        The ensemble size to build.
    M0 : int, optional
        The number of estimators to use for the ECV estimate.    
    kwargs_ensemble : dict, optional
        Additional keyword arguments for the ensemble model.    
    M_max : int, optional
        The maximum ensemble size to consider for the tuned ensemble.
    delta : float, optional
        The suboptimality parameter for the ensemble size tuning by ECV.
    return_df : bool, optional
        If True, returns the results as a pandas.DataFrame object.
    n_jobs : int, optional
        The number of jobs to run in parallel. If -1, all CPUs are used.
    X_val,Y_val : numpy.array, optional
        The validation samples. It may be useful to be used for comparing the 
        performance of ECV with other cross-validation methods that requires sample-splitting.
    '''
    
    keys_to_keep = ['max_samples', 'max_features']

    n_grid = grid.shape[0]
    dtypes = grid.dtypes
    valid = X_val is not None and Y_val is not None
    n_res = 2*M if valid else M
    res_risk = np.full((n_grid,n_res), np.inf)
    
    for i in range(n_grid):
        params = grid.iloc[i].to_dict()
        params_ensemble = {k: dtypes[k].type(v) for k, v in params.items() if k in keys_to_keep}
        params_regr = {k: dtypes[k].type(v) for k, v in params.items() if k not in keys_to_keep}
        
        _, res = comp_empirical_ecv(
            X_train, Y_train, 
            regr, {**kwargs_regr, **params_regr}, 
            M, M0, {**kwargs_ensemble, **params_ensemble}, n_jobs, X_val, Y_val, **kwargs
        )
        res_risk[i, :] = np.r_[res]

    if return_df:
        cols = np.char.add(['risk_ecv-']*M, np.char.mod('%d', 1+np.arange(M)))
        if valid:
            cols = np.append(cols, np.char.add(['risk_test-']*M, np.char.mod('%d', 1+np.arange(M))))
        res_ecv = pd.concat([grid, 
                             pd.DataFrame(res_risk, columns=cols)
                             ] ,axis=1)
    else:
        if valid:            
            res_ecv = (res_risk[:,:M], res_risk[:,M:])
        else:
            res_ecv = res_risk

    j = np.nanargmin(2 * res_risk[:,1] - res_risk[:,0])
    best_params_ = grid.iloc[j].to_dict()
    if delta==0.:
        M_hat = np.inf
    else:
        M_hat = int(np.ceil(2 / delta * (res_risk[j,0] - res_risk[j,1])))
    best_n_estimators_ = np.minimum(M_hat, M_max)

    info_ecv = {
        'delta': delta,
        'best_params': best_params_,
        'best_n_estimators': best_n_estimators_,
        'M_max':M_max
    }
    return res_ecv, info_ecv
    
    





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