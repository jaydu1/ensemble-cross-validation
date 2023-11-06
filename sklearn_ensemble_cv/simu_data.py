import numpy as np
import scipy as sp
from scipy.linalg import sqrtm, toeplitz, block_diag
from scipy.sparse.linalg import eigsh



def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):
    '''
    Parameters
    ----------
    sigma_sq: float
        scalar_like,
    first_column_except_1: array
        1d-array, except diagonal 1.
    
    Returns
    ----------
        2d-array with dimension (len(first_column)+1, len(first_column)+1)
    '''
    first_column = np.append(1, first_column_except_1)
    cov_mat = sigma_sq * toeplitz(first_column)
    return cov_mat


def ar1_cov(rho, n, sigma_sq=1):
    """
    Parameters
    ----------    
    rho : float
        scalar, should be within -1 and 1.
    n : int
        scalar, number of rows and columns of the covariance matrix.
    sigma_sq : float
        scalar, multiplicative factor of the covariance matrix.
    
    Returns
    ----------
        2d-matrix of (n,n)
    """
    if rho != 0.:
        rho_tile = rho * np.ones([n - 1])
        first_column_except_1 = np.cumprod(rho_tile)
        cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    else:
        cov_mat = np.identity(n)
    return cov_mat


def block_ar1_cov(rhos, n):
    """
    Parameters
    ----------
    rhos : float
        array, should be within -1 and 1.
    n : int
    
    Returns
    ----------
        2d-matrix of (n,n)
    """
    n_block = len(rhos)
    s_block = n // n_block
    covs = []
    for i in range(n_block):
        ns = s_block if i < n_block - 1 else n - s_block * (n_block - 1)
        covs.append(ar1_cov(rhos[i], ns))
    cov_mat = block_diag(*covs)
    return cov_mat



def generate_data(
    n, p, coef='random', func='linear',
    rho_ar1=0., sigma=1, df=np.inf, n_test=1000, sigma_quad=1.,
):
    """
    Parameters
    ----------
    n : int
        sample size
    p : int
        number of features
    coef : str or array-like
        If 'sorted', the coefficients are sorted in descending order of absolute value.
        If 'random', the coefficients are randomly generated from a uniform distribution.
        If an array-like object, the coefficients are set to the given values.
    func : str
        The functional form of the relationship between the features and the response.
        If 'linear', the response is a linear combination of the features.
        If 'nonlinear', the response is a nonlinear function of the features.
    rho_ar1 : float or array-like
        If a scalar, the AR(1) correlation coefficient for the features.
        If an array-like object, the AR(1) correlation coefficients for each block of features.
    sigma : float
        The standard deviation of the noise.
    df : float
        The degrees of freedom for the t-distribution used to generate the noise.
        If np.inf, the noise is generated from a normal distribution.
    n_test : int
        The number of test samples to generate.
    sigma_quad : float
        The standard deviation of the quadratic term in the nonlinear function.
    
    Returns
    -------
    X_train : ndarray of shape (n, p)
        The training features.
    y_train : ndarray of shape (n,)
        The training response.
    X_test : ndarray of shape (n_test, p)
        The test features.
    y_test : ndarray of shape (n_test,)
        The test response.
    """
        
    if np.isscalar(rho_ar1):
        Sigma = ar1_cov(rho_ar1, p)
    else:
        Sigma = block_ar1_cov(rho_ar1, p)

    if df==np.inf:
        Z = np.random.normal(size=(n,p))
        Z_test = np.random.normal(size=(n_test,p))
    else:
        Z = np.random.standard_t(df=df, size=(n,p)) / np.sqrt(df / (df - 2))
        Z_test = np.random.standard_t(df=df, size=(n_test,p)) / np.sqrt(df / (df - 2))
        
    Sigma_sqrt = sqrtm(Sigma)
    X = Z @ Sigma_sqrt
    X_test = Z_test @ Sigma_sqrt
    
    if sigma<np.inf:
        if coef.startswith('eig'):
            top_k = int(coef.split('-')[1])
            _, beta0 = eigsh(Sigma, k=top_k)
            
            beta0 = np.mean(beta0, axis=-1)
            if np.isscalar(rho_ar1):
                rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
            else:
                rho2 = np.linalg.norm(beta0)**2
        elif coef=='sorted':
            beta0 = np.random.normal(size=(p,))
            beta0 = beta0[np.argsort(-np.abs(beta0))]
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.        
        elif coef=='uniform':
            beta0 = np.ones(p,) / np.sqrt(p)
            rho2 = 1.
        elif coef=='random':
            beta0 = np.random.normal(size=(p,))
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.
        elif coef.startswith('sparse'):
            s = int(coef.split('-')[1])
            beta0 = np.zeros(p,)
            beta0[:s] = 1/np.sqrt(s)
            rho2 = 1.
            
    else:
        rho2 = 0.
        beta0 = np.zeros(p)

    Y = X@beta0[:,None]   
    Y_test = X_test@beta0[:,None]
    
    if func=='linear':
        pass
    elif func=='quad':
        Y += sigma_quad * (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += sigma_quad * (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    elif func=='tanh':
        Y = np.tanh(Y)
        Y_test = np.tanh(Y_test)
    else:
        raise ValueError('Not implemented.')

    if sigma>0.:
        if df==np.inf:
            Y = Y + sigma * np.random.normal(size=(n,1))            
            Y_test += sigma * np.random.normal(size=(n_test,1))
        else:
            Y = Y + np.random.standard_t(df=df, size=(n,1))
            Y_test += np.random.standard_t(df=df, size=(n_test,1))
            sigma = np.sqrt(df / (df - 2))
    else:
        sigma = 0.

    return Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma**2