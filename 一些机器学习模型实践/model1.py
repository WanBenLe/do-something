# coding utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.feature_selection import SelectFpr, f_regression
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import keras

# 数据文件
d1 = pd.read_csv('well.csv').values
print(d1[1,:])
# 收益率
Y = d1[:, 0]
Y = scale(Y)
print(Y)
# X中的数字
d3 = d1[:, 3:12]
# 标准化
d3 = scale(d3, axis=0)
# X中的非数字
d4 = d1[:, 12:]
# 哑变量处理
enc = OneHotEncoder()
enc.fit(d4)
d4 = enc.transform(d4).toarray()
# 合并成X
X = np.hstack((d3, d4))

# model1 Lasso
clf = GridSearchCV(Lasso(max_iter=10000), cv=5,
                   param_grid={'alpha': [1e-5, 1e-4, 1e-3, 1e-2]})
clf.fit(X, Y)
Y_pred = clf.predict(X)
Y_pred[Y_pred > 0] = 1
Y_pred[Y_pred < 0] = 0
r2 = np.hstack((d1[:, 1].reshape(-1, 1), Y_pred.reshape(-1, 1)))
print(r2)
pd.DataFrame(r2, columns=['ID', 'BUY']).to_csv('model1.csv')

# model2 Ridge
clf = GridSearchCV(Ridge(), cv=5,
                   param_grid={'alpha': [1e-5, 1e-4, 1e-3, 1e-2]})
clf.fit(X, Y)
Y_pred = clf.predict(X)
Y_pred[Y_pred > 0] = 1
Y_pred[Y_pred < 0] = 0
r1 = np.hstack((d1[:, 1].reshape(-1, 1), Y_pred.reshape(-1, 1)))
print(r1)
pd.DataFrame(r1, columns=['ID', 'BUY']).to_csv('model2.csv')

# model3 XGBoost
Y = Y.reshape(-1, 1)
all_d = np.hstack((Y, X))
np.random.shuffle(all_d)
# 0.7是训练集的比例
numx = int(all_d.shape[0] * 0.7)
train = all_d[0:numx, :]
test = all_d[numx:, :]
dtrain = xgb.DMatrix(train[:, 1:], label=train[:, 0])
dtest = xgb.DMatrix(test[:, 1:], label=test[:, 0])
evallist = [(dtest, 'eval'), (dtrain, 'train')]
param = {'max_depth': 2, 'eta': 1, 'silent': 1}
param['nthread'] = 4
param['eval_metric'] = 'rmse'
num_round = 100
bst = xgb.train(param, dtrain, num_round, evallist)
bst.save_model('model3.model')

dtest = xgb.DMatrix(X)
Y_pred = bst.predict(dtest)
Y_pred[Y_pred > 0] = 1
Y_pred[Y_pred < 0] = 0
r3 = np.hstack((d1[:, 1].reshape(-1, 1), Y_pred.reshape(-1, 1)))
print(r3)
pd.DataFrame(r3, columns=['ID', 'BUY']).to_csv('model3.csv')

# model4 SVM
clf = SVR(gamma='auto')
clf.fit(train[:, 1:], train[:, 0])
Y_pred = clf.predict(X)
Y_pred[Y_pred > 0] = 1
Y_pred[Y_pred < 0] = 0
r4 = np.hstack((d1[:, 1].reshape(-1, 1), Y_pred.reshape(-1, 1)))
print(r4)
pd.DataFrame(r4, columns=['ID', 'BUY']).to_csv('model4.csv')

# model5 LightGBM
# 根据F test筛选变量
X = SelectFpr(f_regression, alpha=0.01).fit_transform(X, Y.reshape(-1, 1))
Y = Y.reshape(-1, 1)
all_d = np.hstack((Y, X))
np.random.shuffle(all_d)
# 0.7是训练集的比例
numx = int(all_d.shape[0] * 0.7)
train = all_d[0:numx, :]
test = all_d[numx:, :]
dtrain = lgb.Dataset(train[:, 1:], label=train[:, 0])
dtest = lgb.Dataset(test[:, 1:], label=test[:, 0])
num_round = 100
del param
param = {'max_depth': 2, 'num_leaves': 4, 'num_trees': 1, 'objective': 'regression_l1'}
param['metric'] = ['l1']

bst = lgb.train(param, dtrain, num_round)
bst.save_model('model4.txt')
Y_pred = bst.predict(X)
Y_pred[Y_pred > 0] = 1
Y_pred[Y_pred < 0] = 0
r5 = np.hstack((d1[:, 1].reshape(-1, 1), Y_pred.reshape(-1, 1)))
pd.DataFrame(r5, columns=['ID', 'BUY']).to_csv('model5.csv')



# 6
model = Sequential()
model.add(Dense(np.shape(X)[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mae'])
model.fit(train[:, 1:], train[:, 0], epochs=10, batch_size=5)
score = model.evaluate(test[:, 1:], test[:, 0], batch_size=5)
Y_pred = clf.predict(X)
Y_pred[Y_pred > 0] = 1
Y_pred[Y_pred < 0] = 0
r6 = np.hstack((d1[:, 1].reshape(-1, 1), Y_pred.reshape(-1, 1)))
print(r5)
pd.DataFrame(r6, columns=['ID', 'BUY']).to_csv('model6.csv')
