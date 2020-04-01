import pandas as pd
from sklearn.linear_model import Lasso
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import warnings

'''
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
'''
warnings.filterwarnings("ignore")
data0 = pd.read_csv('cndata.csv')

data0 = data0[data0['codate'] > 200400]
data1 = data0[data0['codate'] <= 201400]
d1 = data1[['codate', 'retx']]
d1.index = range(len(d1['codate']))
d2 = data1.drop(columns=['retx'])

data2 = data0[data0['codate'] > 201400]
d3 = data2[['codate', 'retx']]
d3.index = range(len(d3['codate']))
d4 = data2.drop(columns=['retx'])

date_1 = list(set(d2['codate'].values))
date_1.sort()
date_2 = list(set(d4['codate'].values))
date_2.sort()


def tas(data1, data2, date_x):
    for i in range(len(date_x)):
        a = data1[data1['codate'] == date_x[i]]
        b = np.mean(data1['retx'].values)
        a = data2[data2['codate'] == date_x[i]].values.T[1:]
        for j in range(len(a)):
            t = np.array(np.mean(a[j]))
            if j == 0:
                y1 = t
            else:
                y1 = np.hstack((y1, t))
        if i == 0:
            y2 = y1
            x2 = b
        else:
            y2 = np.vstack((y2, y1))
            x2 = np.vstack((x2, b))
    x2 = x2[:-1]
    y2 = y2[1:]
    return x2, y2


r1, x1 = tas(d1, d2, date_1)
r2, x2 = tas(d3, d4, date_2)
x1[np.isnan(x1)] = 0.0
x2[np.isnan(x2)] = 0.0
r1[np.isnan(r1)] = 0.0
r2[np.isnan(r2)] = 0.0

alpha_ridge = [1e-15,1e-13,1e-11,1e-9,1e-7]
coeffs = {}

for alpha in alpha_ridge:
    r = Lasso(alpha=alpha, normalize=True, max_iter=1000000)
    r = r.fit(x1, r1)

grid_search = GridSearchCV(Lasso(alpha=1, normalize=False), scoring='neg_mean_squared_error',
                           param_grid={'alpha': alpha_ridge}, cv=10, n_jobs=-1)
grid_search.fit(x1, r1)
alpha = alpha_ridge
rmse = list(np.sqrt(-grid_search.cv_results_['mean_test_score']))
print(rmse)
plt.figure(figsize=(6, 5))
lasso_cv = pd.Series(rmse, index=alpha)
lasso_cv.plot(title="Validation - LASSO", logx=True)
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

lasso = Lasso(alpha=0, normalize=False, max_iter=1e6)
lasso = lasso.fit(x1, r1)
coef = pd.Series(lasso.coef_)
print(coef)