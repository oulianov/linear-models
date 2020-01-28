"""
Multi target regression with Linear model
Implementation of 'Adaptative multivariate Ridge regression' (1980) by P.J. Brown and J.V. Zidek 
Code by Nicolas Oulianov (2019)
"""

import numpy as np 

## OLS estimation
# Beta is an estimator of C
def ols_regression(X, y):
    """X : shape (n,p) observations
    y : shape (n,q) target
    Returns beta such as X.dot(beta) is close to y"""
    if X.shape[0] != y.shape[0]:
        raise Exception
    
    # Note : X must be of rank < min(p,q)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    return beta


## Ridge estimation
# The ridge matrix
def ridge_regression(X, y, K=None):
    """X : shape (n,p) observations
    y : shape (n,q) target
    K : shape (q,q) the Ridge matrix
    Returns beta such as X.dot(beta) is close to y, with coefficients of beta more stable."""
    if type(K) is int or type(K) is float or (type(K) is np.array and K.shape==(1,1)):
        K = np.ones((q,q))*K
    elif K is None:
        K = np.ones((q,q))

    if X.shape[0] != y.shape[0]:
        raise ValueError('different number of observation in X and y: {} != {}'.format(X.shape[0], y.shape[0])) 

    p = X.shape[1]
    q = y.shape[1]

    if K.shape != (q,q):
        raise ValueError('expected a matrix K of shape ({},{}) and not {}'.format(q,q,K.shape))
    
    # Compute the OLS estimator
    beta = ols_regression(X, y)

    I_p = np.eye(p,p)
    I_q = np.eye(q,q)

    beta_K = np.linalg.inv(np.kron(X.T.dot(X), I_q) + np.kron(I_p, K)).dot(np.kron(X.T.dot(X), I_q)).dot(beta.flatten('C'))
    beta_K = beta_K.reshape((p,q))

    return beta_K

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    q = 20 # output size
    p = 80 # input size
    n = 300 # number of observations

    # Build data
    # Observation vector. This is the data we have from the field.
    X = np.random.normal(loc=10, scale=3, size=(n, p))
    # Coefficient vector. This is the matrix we're trying to estimate.
    C = np.random.normal(loc=5, size=(p,q))
    # To induce some colinearity, we make some columns equals to a multiple of others.
    C[:-5] = C[5:]*5
    # Error. This is what makes predictions uncertain. 
    epsilon = np.random.normal(size=(n, q))*2
    # Target. This is what we're trying to predict. 
    y = np.dot(X, C) + epsilon

    # Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    for u in [0.1, 1, 10, 100]:
        K = np.ones((q,q))*u
        beta_K = ridge_regression(X_train, y_train, K)
        print('Ridge parameter :', u)
        print('Coefficient accuracy : {}%'.format(100 - round(100*np.linalg.norm(beta_K-C)/np.linalg.norm(C), 4)))
        print('Prediction risk : +/- {}%'.format(round(100*np.std(y_test - np.dot(X_test, beta_K))/np.mean(y_test), 4)))

