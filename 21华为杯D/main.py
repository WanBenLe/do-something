import pandas as pd
import numpy as np
from DSLasso import dslasso
from sklearn.preprocessing import scale, LabelEncoder
import pickle
from statsmodels.api import OLS, add_constant
from xgboost.sklearn import XGBRegressor, XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from pickle import load, dump
import seaborn

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
# 第一问
# X
# d1 = pd.read_excel('Molecular_Descriptor.xlsx')
# pickle.dump(d1,open('d1.pkl','wb'))
d1 = pickle.load(open('d1.pkl', 'rb'))

'''
fit_data = d1.values[:, 1:].astype(float)
fit_data1 = scale(fit_data)
Set_1, Set_2, Set_all, st, sr = dslasso(fit_data1, y)
# DS-Lasso选择的第一阶段变量
namex = np.array(list(d1.columns)[1:])[Set_all == 1].tolist()
print(sum(Set_1))
print(sum(Set_2))
x = fit_data1[:, Set_1]
z = fit_data1[:, Set_2 == 1]
X = (np.hstack((x, z))).astype(float)
ols = OLS(y, X).fit()
print(ols.summary())
#OLS的结果
result_sheet = np.zeros((X.shape[1], 3), dtype=object)
result_sheet[:, 0] = namex
result_sheet[:, 1] = ols.params
result_sheet[:, 2] = ols.pvalues
result_sheet = result_sheet[np.argsort(result_sheet[:, 2])]
result1 = pd.DataFrame(result_sheet, columns=['分子描述符', '系数','P值'])
result1.to_csv('第一问结果.csv',encoding='utf_8_sig')
selectresult = result_sheet[0:20]
'''
cols = list(d1.columns)[1:]
# 第二问
sel = pd.read_csv('第一问结果.csv').values[:20, 1]
Xall = scale(d1[sel].values)

for i in range(20):
    temp = d1[sel].iloc[:, i]
    seaborn.distplot(temp, hist=True, kde=True)
    plt.title(cols[i] + '核密度和直方图')
    plt.savefig(cols[i] + '核密度和直方图''.png')
print(1)
a1 = d1[sel].corr()
a1.to_excel('corr.xlsx', index=False)
seaborn.heatmap(a1, vmin=0, vmax=1, center=0)
plt.title("heatmap")
plt.xlabel("x_ticks")
plt.ylabel("y_ticks")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(Xall, y, test_size=0.4)

mdl = SVR(C=9).fit(X_train, y_train.reshape(-1, 1))
dump(mdl, open('svr.pkl', 'wb'))
trainx = mdl.predict(X_train)
per = mdl.predict(X_test)
trmae = np.round(mean_absolute_error(y_train, trainx), 4)
print('train MAE:', trmae)

temae = np.round(mean_absolute_error(y_test, per), 4)
print('test MAE:', temae)
d2 = pd.read_excel('Molecular_Descriptor.xlsx', sheet_name='test')[sel].values
y1 = mdl.predict(scale(d2))
y2 = -np.log(y1 * 10 ** -9)

yx = pd.read_excel('ERα_activity.xlsx', sheet_name='test')
yx['IC50_nM'] = y1
yx['pIC50'] = y2
yx.to_excel('问题2结果.xlsx', index=False)

# 第三问
Xall = d1.values[:, 1:].astype(float)
xmean = np.mean(Xall, axis=0)
xstd = np.std(Xall, axis=0)
Xall = scale(Xall)
d2 = pd.read_excel('ADMET.xlsx').values[:, 1:].astype(int)
d3 = pd.read_excel('ADMET.xlsx', sheet_name='test')
cols = list(d3.columns)[1:]
d4 = pd.read_excel('Molecular_Descriptor.xlsx', sheet_name='test').values[:, 1:].astype(float)
# d4=(d4-xmean)/xstd
d4 = scale(d4)
mdlS1 = []
mdlS2 = []
mdlS3 = []
for i in range(5):
    print('变量', cols[i])
    yall = d2[:, i]
    yall = LabelEncoder().fit_transform(yall).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(Xall, yall, test_size=0.4)
    mdl1 = GaussianNB().fit(X_train, y_train)
    mdlS1.append(mdl1)
    tr1 = mdl1.predict(X_train)
    tt1 = mdl1.predict(X_test)
    print('NB模型')
    print('训练ACC', accuracy_score(y_train, tr1))
    print('训练F1', accuracy_score(y_test, tt1))
    print('测试ACC', f1_score(y_train, tr1))
    print('测试F1', f1_score(y_test, tt1))
    mdl2 = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=3, min_child_weight=0.1, gamma=0, reg_alpha=1,
                         colsample_bytree=0.2, subsample=0.5, objective='binary:logistic', nthread=4,
                         ).fit(X_train, y_train)
    mdlS2.append(mdl2)
    tr2 = mdl2.predict(X_train)
    tt2 = mdl2.predict(X_test)
    print('XGBoost模型')
    print('训练ACC', accuracy_score(y_train, tr2))
    print('训练F1', accuracy_score(y_test, tt2))
    print('测试ACC', f1_score(y_train, tr2))
    print('测试F1', f1_score(y_test, tt2))
    ptr1 = mdl1.predict_proba(X_train)[:, 1].reshape(-1, 1)
    ptt1 = mdl1.predict_proba(X_test)[:, 1].reshape(-1, 1)
    ptr2 = mdl2.predict_proba(X_train)[:, 1].reshape(-1, 1)
    ptt2 = mdl2.predict_proba(X_test)[:, 1].reshape(-1, 1)
    X1 = np.hstack((ptr1, ptr2))
    X2 = np.hstack((ptt1, ptt2))
    mdl3 = LogisticRegressionCV(cv=5).fit(X1, y_train)
    mdlS3.append(mdl3)
    tr3 = mdl3.predict(X1)
    tt3 = mdl3.predict(X2)
    print('集成LR模型')
    print('训练ACC', accuracy_score(y_train, tr3))
    print('训练F1', accuracy_score(y_test, tt3))
    print('测试ACC', f1_score(y_train, tr3))
    print('测试F1', f1_score(y_test, tt3))

    p1 = mdl1.predict_proba(d4)[:, 1].reshape(-1, 1)
    p2 = mdl2.predict_proba(d4)[:, 1].reshape(-1, 1)

    d3[cols[i]] = mdl3.predict(np.hstack((p1, p2)))

# d3.to_excel('问题3结果.xlsx')


dump(mdlS1, open('mdlS1.pkl', 'wb'))
dump(mdlS2, open('mdlS2.pkl', 'wb'))
dump(mdlS3, open('mdlS3.pkl', 'wb'))

# 第四问
svr = load(open('svr.pkl', 'rb'))
mdlS1 = load(open('mdlS1.pkl', 'rb'))
mdlS2 = load(open('mdlS2.pkl', 'rb'))
mdlS3 = load(open('mdlS3.pkl', 'rb'))
sel = pd.read_csv('第一问结果.csv').values[:20, 1]
cols = np.array(list(d1.columns)[1:])
colsw = []
for i in range(len(sel)):
    colsw.append(np.argwhere(cols == sel[i])[0][0])

Xall2 = scale(d1.values[:, 1:])
xmean1 = np.mean(Xall2, axis=0)
xstd1 = np.std(Xall2.astype(float), axis=0)

xmean2 = np.mean(d1.values[:, 1:], axis=0)
xstd2 = np.std(d1.values[:, 1:].astype(float), axis=0)


class huilangopt():
    # 优化的目标函数
    def optfun(self):
        x1 = self.Positions[:, self.colsw]
        y1 = self.svr.predict(x1)
        y2 = np.zeros_like(y1)
        for i in range(5):
            p1 = self.mdlS1[i].predict_proba(self.Positions)[:, 1].reshape(-1, 1)
            p2 = self.mdlS2[i].predict_proba(self.Positions)[:, 1].reshape(-1, 1)
            p3 = self.mdlS3[i].predict(np.hstack((p1, p2)))
            y2 = y2 + p3
        self.fitness = np.zeros_like(y1)
        self.fitness[y2 < 3] = 10 * 10
        self.fitness -= y1

    def initialization(self):
        Boundary_no = self.ub.shape[0]

        if Boundary_no == 1:
            self.Positions = np.random.rand((self.SearchAgents_no, self.dim)) * (self.ub - self.lb) + self.lb

        if Boundary_no > 1:
            for i in range(self.dim):
                ub_i = self.ub[i]
                lb_i = self.lb[i]
                temp = np.random.rand(self.SearchAgents_no, 1) * (ub_i - lb_i) + lb_i
                if i == 0:
                    self.Positions = temp
                else:
                    self.Positions = np.hstack((self.Positions, temp))

    def __init__(self, colsw, svr, mdlS1, mdlS2, mdlS3, xmean1, xmean2, xstd1, xstd2):

        self.colsw = colsw
        self.svr = svr
        self.mdlS1 = mdlS1
        self.mdlS2 = mdlS2
        self.mdlS3 = mdlS3
        self.xmean2 = xmean2
        self.xstd1 = xstd1
        self.xmean1 = xmean1
        self.xstd2 = xstd2
        self.SearchAgents_no = 100  # 狼群数量，Number of search agents
        Max_iteration = 20  # 最大迭代次数，Maximum numbef of iterations
        self.dim = 729  # 此例需要优化两个参数c和g，number of your variables
        self.lb = np.ones((self.dim)) * 0  # 参数取值下界
        self.ub = np.ones((self.dim)) * 250
        # v = 5# SVM Cross Validation参数,默认为5

        # initialize alpha, beta, and delta_pos
        Alpha_pos = np.zeros((1, self.dim)).reshape(-1)  # 初始化Alpha狼的位置
        Alpha_score = np.inf  # 初始化Alpha狼的目标函数值，change this to -inf for maximization problems

        Beta_pos = np.zeros((1, self.dim)).reshape(-1)  # beta狼的位置
        Beta_score = np.inf  # 初始化Beta狼的目标函数值，change this to -inf for maximization problems

        Delta_pos = np.zeros((1, self.dim)).reshape(-1)  # 初始化Delta狼的位置
        Delta_score = np.inf  # 初始化Delta狼的目标函数值，change this to -inf for maximization problems

        # Initialize the positions of search agents
        self.initialization()

        Convergence_curve = np.zeros((Max_iteration, 1))

        # Main loop主循环
        for l in range(Max_iteration):
            for i in range(len(self.Positions)):

                # Return back the search agents that go beyond the boundaries of the search space
                # 若搜索位置超过了搜索空间，需要重新回到搜索空间
                Flag4ub = self.Positions[i, :] > self.ub
                Flag4lb = self.Positions[i, :] < self.lb
                # 若狼的位置在最大值和最小值之间，则位置不需要调整，若超出最大值，最回到最大值边界；
                # 若超出最小值，最回答最小值边界
                self.Positions[i, :] = (self.Positions[i, :] * (
                    ~(Flag4ub + Flag4lb))) + self.ub * Flag4ub + self.lb * Flag4lb  # ~表示取反
                self.optfun()

                # Update Alpha, Beta, and Delta
                if self.fitness[i] < Alpha_score:  # 如果目标函数值小于Alpha狼的目标函数值
                    Alpha_score = self.fitness[i]  # 则将Alpha狼的目标函数值更新为最优目标函数值，Update alpha
                    Alpha_pos = self.Positions[i, :]  # 同时将Alpha狼的位置更新为最优位置

                if (self.fitness[i] > Alpha_score) and (self.fitness[i] < Beta_score):  # 如果目标函数值介于于Alpha狼和Beta狼的目标函数值之间
                    Beta_score = self.fitness[i]  # 则将Beta狼的目标函数值更新为最优目标函数值，Update beta
                    Beta_pos = self.Positions[i, :]  # 同时更新Beta狼的位置

                if (self.fitness[i] > Alpha_score) and (self.fitness[i] > Beta_score) and (
                        self.fitness[i] < Delta_score):  # 如果目标函数值介于于Beta狼和Delta狼的目标函数值之间
                    Delta_score = self.fitness[i]  # 则将Delta狼的目标函数值更新为最优目标函数值，Update delta
                    Delta_pos = self.Positions[i, :]  # 同时更新Delta狼的位置

            a = 2 - l * ((2) / Max_iteration)  # 对每一次迭代，计算相应的a值，a decreases linearly fron 2 to 0

            # Update the Position of search agents including omegas
            for i in range(len(self.Positions)):  # 遍历每个狼
                for j in range(self.Positions.shape[1]):  # 遍历每个维度

                    # 包围猎物，位置更新

                    r1 = np.random.rand()  # r1 is a random number in [0,1]
                    r2 = np.random.rand()  # r2 is a random number in [0,1]

                    A1 = 2 * a * r1 - a  # 计算系数A，Equation (3.3)
                    C1 = 2 * r2  # 计算系数C，Equation (3.4)

                    # Alpha狼位置更新
                    D_alpha = np.abs(C1 * Alpha_pos[j] - self.Positions[i, j])  # Equation (3.5)-part 1
                    X1 = Alpha_pos[j] - A1 * D_alpha  # Equation (3.6)-part 1

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A2 = 2 * a * r1 - a  # 计算系数A，Equation (3.3)
                    C2 = 2 * r2  # 计算系数C，Equation (3.4)

                    # Beta狼位置更新

                    D_beta = np.abs(C2 * Beta_pos[j] - self.Positions[i, j])  # Equation (3.5)-part 2
                    X2 = Beta_pos[j] - A2 * D_beta  # Equation (3.6)-part 2

                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A3 = 2 * a * r1 - a  # 计算系数A，Equation (3.3)
                    C3 = 2 * r2  # 计算系数C，Equation (3.4)

                    # Delta狼位置更新
                    D_delta = np.abs(C3 * Delta_pos[j] - self.Positions[i, j])  # Equation (3.5)-part 3
                    X3 = Delta_pos[j] - A3 * D_delta  # Equation (3.5)-part 3

                    # 位置更新

                    self.Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7)
            Convergence_curve[l] = Alpha_score
        plt.plot(Convergence_curve)
        plt.title('性质超过3的负活性的优化迭代图')
        plt.show()
        pos = (Alpha_pos * self.xstd2) + self.xmean2
        pd.DataFrame(np.hstack((pos.reshape(-1, 1), cols.reshape(-1, 1))), columns=['取值', '名字']).to_excel('问题4.xlsx')
        print('活性', -Alpha_score)
        print('负对数活性', -np.log(-Alpha_score * 10 ** -9))
        # print('最优参数',pos)

        print(1)


huilangopt(colsw, svr, mdlS1, mdlS2, mdlS3, xmean1, xmean2, xstd1, xstd2)
print(1)
