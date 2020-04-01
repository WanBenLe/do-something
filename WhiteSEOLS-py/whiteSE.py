import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import f, t

aaa1 = np.random.normal(size=(1,30))
aaa2 = np.random.normal(size=(30, 3))
print(aaa1[0].shape)

def OLS_robustSE(x, y):
    matrix_a = np.hstack((x, np.ones((len(x), 1))))
    _para = np.linalg.lstsq(matrix_a, y.T, rcond=-1)[0]
    c=np.multiply(matrix_a, _para.T)
    c = np.sum(c, axis=1)
    _resid = (y - c) ** 2
    rss = np.sum(_resid)
    ess = np.sum((y - np.mean(y)) ** 2)
    _r2 = 1 - (rss / (rss + ess))
    n = y.shape[1]
    k = y.shape[0]
    _adj_r2 = 1 - (rss / (n - k - 1)) / ((rss + ess) / (n - 1))
    _fstat = (ess / k) / (rss / (n - k - 1))
    _fp = f.sf(np.abs(_fstat), k, n - k - 1) * 2
    xxt = np.dot(matrix_a, matrix_a.T)
    resxxt = np.dot(_resid, _resid.T)
    print(resxxt)
    resstd = np.std((y - c), ddof=(1 + k))
    _result = np.zeros((5, len(_para)))
    _result[0, :] = _para.reshape(1,-1)
    for j in range(len(_para)):
        _result[1, j] = _para[j] / (xxt[j, j] * resstd) ** 0.5
        _result[2, j] = t.sf(np.abs(_result[1, j]), n - k - 1) * 2
        _result[3, j] = _para[j] / (resxxt[j, j] * resstd) ** 0.5
        _result[4, j] = t.sf(np.abs(_result[3, j]), n - k - 1) * 2

    print("#####coef,t-stat,p-value#####")
    print(_result)
    print("R^2= ", _r2)
    print("Adj. R^2= ", _adj_r2)
    print("para ", _para)
    print("F-stat: ", _fstat)
    print("p-value: ", _fp)
    print("#############################")
    return _result


OLS_robustSE(aaa2, aaa1)
