from typing import List
from sklearn.ensemble import BaggingRegressor
from sklearn_ensemble_cv.utils import risk_estimate, degree_of_freedom

from joblib import Parallel, delayed
n_jobs = 16

import numpy as np
import pandas as pd
from functools import reduce
import itertools

import warnings
warnings.filterwarnings('ignore') 



class Ensemble(BaggingRegressor):
    '''
    Ensemble class is built on top of sklearn.ensemble.BaggingRegressor.
    It provides additional methods for computing ECV estimates.
    '''
    def __init__(self, **kwargs):
        super(BaggingRegressor, self).__init__(**kwargs)


    def predict_individual(self: BaggingRegressor, X: np.ndarray, n_jobs: int=-1) -> np.ndarray:
        '''
        Predicts the target values for the given input data using the provided BaggingRegressor model.

        Parameters
        ----------
            regr : BaggingRegressor
                The BaggingRegressor model to use for prediction.
            X : np.ndarray
                [n, p] The input data to predict target values for.

        Returns:
            Y_hat : np.ndarray
                [n, M] The predicted target values of all $M$ estimators for the input data.
        '''
        with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
            Y_hat = parallel(
                delayed(lambda reg,X: reg.predict(X[:,features]).reshape(-1,1))(reg, X)
                for reg,features in zip(self.estimators_,self.estimators_features_)
            )
        Y_hat = np.concatenate(Y_hat, axis=-1)
        return Y_hat
    

    def compute_risk(self, X, Y, M_test=None, return_df=False, n_jobs=-1, **kwargs_est):
        if M_test is None:
            M_test = self.n_estimators
        if M_test>self.n_estimators:
            raise ValueError('The ensemble size M must be less than or equal to the number of estimators in the BaggingRegressor model.')
        if Y.ndim==1:
            Y = Y[:,None]

        Y_hat = self.predict_individual(X, n_jobs)
        Y_hat = np.cumsum(Y_hat, axis=1) / np.arange(1,M_test+1)
        err_eval = (Y_hat - Y)**2
        risk = risk_estimate(err_eval, axis=0, **kwargs_est)

        if return_df:
            df = pd.DataFrame({'M':np.arange(1,M_test+1), 'risk':risk})
            return df
        else:
            return risk
    
    
    def compute_ecv_estimate(self, X_train, Y_train, M_test=None, M0=None, return_df=False, n_jobs=-1, **kwargs_est):
        '''
        Computes the ECV estimate for the given input data using the provided BaggingRegressor model.

        Parameters
        ----------
        X_train : np.ndarray
            [n, p] The input covariates.
        Y_train : np.ndarray
            [n, ] The target values of the input data.
        M_test : int or np.ndarray
            The maximum ensemble size of the ECV estimate.
        M0 : int, optional
            The number of estimators to use for the OOB estimate. If None, M0 is set to the number of estimators in the BaggingRegressor model.
        return_df : bool, optional
            If True, returns the ECV estimate as a pandas.DataFrame object.
        n_jobs : int, optional
            The number of jobs to run in parallel. If -1, all CPUs are used.
        kwargs_est : dict
            Additional keyword arguments for the risk estimate.

        Returns:
        --------
        risk_ecv : np.ndarray or pandas.DataFrame
            [M_test, ] The ECV estimate for each ensemble size in M_test.
        '''
        if M_test is None:
            M_test = self.n_estimators
        if M0 is None:
            M0 = self.n_estimators
        if M0<2:
            raise ValueError('The ensemble size M or M0 must be at least 2.')
        if np.isscalar(M_test):
            M_test = np.arange(1,M_test+1)
        else:
            M_test = np.array(M_test)
        if Y_train.ndim==1:
            Y_train = Y_train[:,None]
        ids_list = self.estimators_samples_[:M0]
        Y_hat = self.predict_individual(X_train, n_jobs)
        dev_eval = Y_hat - Y_train
        err_eval = dev_eval**2
        
        with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
            res_1 = parallel(
                delayed(lambda j:risk_estimate(
                    np.delete(err_eval[:,j], ids_list[j]), **kwargs_est)
                       )(j)
                for j in np.arange(M0)
            )
            res_2 = parallel(
                delayed(lambda i,j:risk_estimate(
                    np.mean(
                        np.delete(dev_eval[:,[i,j]], 
                                  np.union1d(ids_list[i], ids_list[j]), axis=0), axis=1)**2,
                    **kwargs_est
                ))(i,j)
                for i,j in itertools.combinations(np.arange(M0), 2)
            )
                    
        risk_ecv_1 = np.nanmean(res_1)
        risk_ecv_2 = np.nanmean(res_2)
        risk_ecv = - (1-2/M_test) * risk_ecv_1 + 2*(1-1/M_test) * risk_ecv_2
        if return_df:
            df = pd.DataFrame({'M':M_test, 'estimate':risk_ecv})
            return df
        else:
            return risk_ecv
        

    def compute_gcv_estimate(self, X_train, Y_train, type='full', return_df=False, n_jobs=-1, **kwargs_est):
        '''
        Computes the GCV estimate for the given input data using the provided BaggingRegressor model.

        Parameters
        ----------
        X_train : np.ndarray
            [n, p] The input covariates.
        Y_train : np.ndarray
            [n, ] The target values of the input data.
        type : str, optional
            The type of GCV estimate to compute. Can be either 'full' (the naive GCV using full observations) or 
            'union' (the naive GCV using training observations).
        return_df : bool, optional
            If True, returns the GCV estimate as a pandas.DataFrame object.
        n_jobs : int, optional
            The number of jobs to run in parallel. If -1, all CPUs are used.
        kwargs_est : dict
            Additional keyword arguments for the risk estimate.

        Returns:
        --------
        risk_gcv : np.ndarray or pandas.DataFrame
            [M_test, ] The GCV estimate for each ensemble size in M_test.
        '''
        if self.base_estimator_.__class__.__name__ not in ['Ridge', 'Lasso', 'ElasticNet']:
            raise ValueError('GCV is only implemented for Ridge, Lasso, and ElasticNet regression.')
        if Y_train.ndim==1:
            Y_train = Y_train[:,None]
        M = self.n_estimators
        M_arr = np.arange(M)
        ids_list = self.estimators_samples_
        Y_hat = self.predict_individual(X_train, n_jobs)
        Y_hat = np.cumsum(Y_hat, axis=1) / (M_arr+1)
        n = X_train.shape[0]
        err_eval = (Y_hat - Y_train)**2

        with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
            dof = parallel(
                delayed(lambda j:degree_of_freedom(self.estimators_[j], X_train[ids_list[j]])
                       )(j)
                for j in M_arr
            )
            dof = np.mean(dof)

        if type=='full':
            err_train = np.mean(err_eval, axis=0)
            deno = (1 - dof/n)**2
            risk_gcv = err_train / deno
        else:            
            ids_union_list = [reduce(np.union1d, ids_list[:j+1]) for j in M_arr]
            n_ids = np.array([len(ids) for ids in ids_union_list])
            err_train = np.array([np.mean(err_eval[ids_union_list[j],j]) for j in M_arr])
            deno = (1 - dof/n_ids)**2
            risk_gcv = err_train / deno
        
        if return_df:
            df = pd.DataFrame({'M':M_arr+1, 'estimate':risk_gcv, 'err_train':err_train, 'deno':deno})
            return df
        else:
            return risk_gcv
        
    

    def compute_cgcv_estimate(self, X_train, Y_train, type='full', return_df=False, n_jobs=-1, **kwargs_est):
        '''
        Computes the GCV estimate for the given input data using the provided BaggingRegressor model.

        Parameters
        ----------
        X_train : np.ndarray
            [n, p] The input covariates.
        Y_train : np.ndarray
            [n, ] The target values of the input data.
        type : str, optional
            The type of CGCV estimate to compute. Can be either 'full' (using full observations) or 
            'ovlp' (using overlapping observations).
        return_df : bool, optional
            If True, returns the GCV estimate as a pandas.DataFrame object.
        n_jobs : int, optional
            The number of jobs to run in parallel. If -1, all CPUs are used.
        kwargs_est : dict
            Additional keyword arguments for the risk estimate.

        Returns:
        --------
        risk_gcv : np.ndarray or pandas.DataFrame
            [M_test, ] The CGCV estimate for each ensemble size in M_test.
        '''
        if self.base_estimator_.__class__.__name__ not in ['Ridge', 'Lasso', 'ElasticNet']:
            raise ValueError('GCV is only implemented for Ridge, Lasso, and ElasticNet regression.')
        if Y_train.ndim==1:
            Y_train = Y_train[:,None]
        M = self.n_estimators
        M_arr = np.arange(M)
        ids_list = self.estimators_samples_
        Y_hat = self.predict_individual(X_train, n_jobs)
          
        n, p = X_train.shape
        if self.estimators_[0].fit_intercept:
            p += 1
        phi = p/n
        k = int(n * self.max_samples)
        psi = p/k

        with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
            dof = parallel(
                delayed(lambda j:degree_of_freedom(self.estimators_[j], X_train[ids_list[j]])
                       )(j)
                for j in M_arr
            )
            dof = np.mean(dof)

        err_train = (Y_hat-Y_train)**2
        if type=='full':
            correct_term = np.mean(err_train) / (1 - 2* dof/n + dof**2/(k*n))
        else:
            correct_term = np.mean([np.mean(err_train[ids_list[j],j]) for j in M_arr]) / (1 - dof/n)**2

        Y_hat = np.cumsum(Y_hat, axis=1) / (M_arr+1)
        err_eval = (Y_hat - Y_train)**2
        err_train = np.array([np.mean(err_eval[:,j]) for j in M_arr])
        deno = (1 - dof/n)**2
        C = 1/ (n/dof - 1)**2 / (M_arr+1) * (psi/phi - 1) * correct_term
        risk_cgcv = err_train / deno - C
        
        if return_df:
            df = pd.DataFrame({'M':M_arr+1, 'estimate':risk_cgcv, 'err_train':err_train, 'deno':deno, 'C':C})
            return df
        else:
            return risk_cgcv
    