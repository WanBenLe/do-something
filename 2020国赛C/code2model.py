import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, LatentDirichletAllocation, FactorAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import joblib

df11 = pd.read_csv('企业信息.csv',encoding='ansi')[['企业代号', '信誉评级']]
df12 = pd.read_csv('tezheng1.csv')
data1 = pd.merge(df11, df12, how='inner', left_on=['企业代号'], right_on=['0'])

del data1['企业代号'], data1['0']
print(data1.columns)
data1 = data1.values

temp = data1[:, 1:].astype(float)
temp[np.isnan(temp)] = 0
data1[:, 1:] = temp

df21 = pd.read_csv('无企业信息.csv', encoding='ansi')[['企业代号']]
df22 = pd.read_csv('tezheng2.csv', encoding='ansi')
data2 = pd.merge(df21, df22, how='inner', left_on=['企业代号'], right_on=['0'])

del data2['0']
print(data2.columns)
data2 = data2.values

temp = data2[:, 1:].astype(float)
temp[np.isnan(temp)] = 0
data2[:, 1:] = temp


# 获取所有标签

class ML_class():

    def inF(self):
        x = self.data[:, 1:].copy()
        # 计算分类信息熵的F值
        F = mutual_info_classif(x, self.data[:, 0])
        joblib.dump(F,'inFmodel')
        print('分类信息熵的F值', F)
        # 取F<0.1的数据
        self.data = np.hstack((self.data[:, 0].reshape(-1, 1), x[:, F < 0.1]))
        self.data2 = self.data2[:, F < 0.1]

    def pca(self):
        print('pca处理')
        # PCA降维取解释方差90%的因子
        modelx = PCA(0.9).fit(self.data[:, 1:])
        joblib.dump(modelx, 'PCAmodel')
        self.data[:, 1:] = modelx.transform(self.data[:, 1:])
        self.data2[:, :] = modelx.transform(self.data2[:, :])
        print(self.data.shape)

    def KPCA(self):
        # KPCA取因子的一半
        shape1 = int(self.data[:, 1:].shape[1] / 2)
        self.data[:, 1:] = KernelPCA(n_components=shape1).fit_transform(self.data[:, 1:])

    def LDA(self):
        # LDA取因子的一半
        shape1 = int(self.data[:, 1:].shape[1] / 2)
        self.data[:, 1:] = LatentDirichletAllocation(n_components=shape1).fit_transform(self.data[:, 1:])

    def FAnalysis(self):
        # LDA取因子的一半
        shape1 = int(self.data[:, 1:].shape[1] / 2)
        self.data[:, 1:] = FactorAnalysis(n_components=shape1).fit_transform(self.data[:, 1:])

    def lightgbm(self):
        print('lightgbm')

        lightgbm = lgb.sklearn.LGBMClassifier(class_weight='balanced')
        param_grid = {
            'learning_rate': [0.01, 0.1,0.3, 0.5,0.7],
            'n_estimators': [30, 40, 50,60,70]
        }
        model = GridSearchCV(lightgbm, param_grid)
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, 'LIGHTGBMmodel')

        self.y_pred = model.predict(self.X_test)
        self.y_pred1 = model.predict(self.data2)

    def NormNB(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)

    def __init__(self, data, data2):
        self.data = data
        self.data2 = data2.copy()
        print(self.data.shape)
        print(self.data2.shape)
        ML_class.inF(self)

        ML_class.pca(self)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[:, 1:], self.data[:, 0],
                                                                                test_size=0.4)

        ML_class.lightgbm(self)

        #print('F1:', f1_score(self.y_test, self.y_pred, average='macro'))
        #print('精确度:', accuracy_score(self.y_test, self.y_pred))
        print(self.y_pred1)


forcast=ML_class(data1, data2[:, 1:]).y_pred1.reshape(-1,1)
a=np.hstack((data2[:,0].reshape(-1,1),forcast))
a=np.hstack((a,forcast))




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


pd.DataFrame(a, columns=['企业代号', '信誉评级', '违约率']).to_csv('truedefaultq2.csv', index=False)