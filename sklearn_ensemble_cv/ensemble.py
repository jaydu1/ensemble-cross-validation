from typing import List
from sklearn.ensemble import BaggingRegressor
from sklearn_ensemble_cv.utils import risk_estimate

from joblib import Parallel, delayed
n_jobs = 16

import numpy as np
import pandas as pd
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
                Additional keyword arguments for the ECV estimate.

        Returns:
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
        

    def compute_gcv_estimate(self, X_train, Y_train, return_df=False, n_jobs=-1):
        raise NotImplementedError('GCV is not implemented yet.')
    

    def compute_cgcv_estimate(self, X_train, Y_train, return_df=False, n_jobs=-1):
        raise NotImplementedError('CGCV is not implemented yet.')
    