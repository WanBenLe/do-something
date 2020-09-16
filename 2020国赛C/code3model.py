import pandas as pd
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA, KernelPCA, LatentDirichletAllocation, FactorAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df21 = pd.read_csv('无企业信息.csv', encoding='ansi')[['企业代号']]
df22 = pd.read_csv('Q3tezheng2.csv', encoding='ansi')
data2 = pd.merge(df21, df22, how='inner', left_on=['企业代号'], right_on=['0'])

del data2['0']
print(data2.columns)
data2 = data2.values

temp = data2[:, 1:].astype(float)
temp[np.isnan(temp)] = 0
data2[:, 1:] = temp
data1x=data2.copy()
data2=data2[:,1:]
F = joblib.load('inFmodel')
data2 = data2[:, F < 0.1]
modelx = joblib.load('PCAmodel')
data2[:, :] = modelx.transform(data2[:, :])
model = joblib.load('LIGHTGBMmodel')
forcast = model.predict(data2).reshape(-1, 1)

a = np.hstack((data1x[:, 0].reshape(-1, 1), forcast))
a = np.hstack((a, forcast))


df1_1 = pd.read_csv('企业信息.csv',encoding='ansi')

print(df1_1.head(10))
dfdeal = df1_1.copy()[['企业代号', '信誉评级', '是否违约']].values
dfdeal[(dfdeal[:, -1] == '是'), -1] = 1
dfdeal[(dfdeal[:, -1] == '否'), -1] = 0
comlen = len(dfdeal)

defalutA = np.sum((dfdeal[:, -2] == 'A') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'A')
defalutB = np.sum((dfdeal[:, -2] == 'B') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'B')
defalutC = np.sum((dfdeal[:, -2] == 'C') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'C')
defalutD = np.sum((dfdeal[:, -2] == 'D') & (dfdeal[:, -1] == 1)) / np.sum(dfdeal[:, -2] == 'D')
print('评级为A的违约率', defalutA)
print('评级为B的违约率', defalutB)
print('评级为C的违约率', defalutC)
print('评级为D的违约率', defalutD)

a[(a[:, 1] == 'A'), -1] = defalutA
a[(a[:, 1] == 'B'), -1] = defalutB
a[(a[:, 1] == 'C'), -1] = defalutC
a[(a[:, 1] == 'D'), -1] = defalutD


pd.DataFrame(a, columns=['企业代号', '信誉评级', '违约率']).to_csv('truedefaultq3.csv', index=False)