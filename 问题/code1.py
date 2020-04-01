# coding utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def VAT(R):
    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))
    J = list(range(0, N))
    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)
    I = i[j]
    del J[I]
    y = np.min(R[I, J], axis=0)
    j = np.argmin(R[I, J], axis=0)
    I = [I, J[j]]
    J = [e for e in J if e != J[j]]
    C = [1, 1]
    for r in range(2, N - 1):
        y = np.min(R[I, :][:, J], axis=0)
        i = np.argmin(R[I, :][:, J], axis=0)
        j = np.argmin(y)
        y = np.min(y)
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
    C.extend([i[j]])
    y = np.min(R[I, :][:, J], axis=0)
    i = np.argmin(R[I, :][:, J], axis=0)
    I.extend(J)
    C.extend(i)
    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx
    RV = R[I, :][:, I]
    return RV.tolist(), C, I


def my_entropy(probs):
    return -probs.dot(np.log2(probs))


def mutual_info(X, Y):
    df = pd.DataFrame.from_dict({'X': X, 'Y': Y})
    Hx = my_entropy(df.iloc[:, 0].value_counts(normalize=True, sort=False))
    Hy = my_entropy(df.iloc[:, 1].value_counts(normalize=True, sort=False))
    counts = df.groupby(["X", "Y"]).size()
    probs = counts / counts.values.sum()
    H_xy = my_entropy(probs)
    # Mutual Information
    I_xy = Hx + Hy - H_xy
    MI = I_xy
    NMI = I_xy / min(Hx, Hy)
    return NMI


# Q1.1
df0 = pd.read_csv('UCI_Credit_Card_Modified.csv', index_col='ID')
data = np.zeros((len(df0.index), 1))
df0.insert(2, 'graduate_school', data)
df0.insert(3, 'university', data)
df0.insert(4, 'high_school', data)
# label2num
for i in df0.index:
    if df0['education'][i] == 1:
        df0.loc[i, ['graduate_school']] = 1
    elif df0['education'][i] == 2:
        df0.loc[i, ['university']] = 1
    elif df0['education'][i] == 3:
        df0.loc[i, ['high_school']] = 1
df0 = df0.drop(['education'], axis=1)
print(df0.head(2))

X_textx = ['limit_bal', 'is_male', 'graduate_school', 'university', 'high_school', 'is_married', 'age', 'pay_1',
           'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5',
           'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

# Q1.2
X = df0.loc[:, ['limit_bal', 'is_male', 'graduate_school', 'university', 'high_school', 'is_married', 'age',
                'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3',
                'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5',
                'pay_amt6']]
Y = df0.loc[:, ['label']]
sc_X = StandardScaler()
X_scaled = sc_X.fit_transform(X)
print('Shape: ', X_scaled.shape)
print('Min: ', round(np.min(X_scaled), 4))
print('Max: ', round(np.max(X_scaled), 4))
print('Average:: ', round(np.mean(X_scaled), 4))
print('Standard Deviation:: ', round(np.std(X_scaled), 4))

# Q1.3
pca = PCA(n_components=2)
X_reduced1 = pca.fit(X).transform(X)
X_reduced2 = pca.fit(X_scaled).transform(X_scaled)

plt.figure(figsize=(13, 10))
plt.subplot(211)
plt.scatter(X_reduced1[:, 0], X_reduced1[:, 1], color=['red', 'blue'], alpha=.8, lw=2)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Dimension reduction without feature scaling')
plt.xlabel("C1,Explained Var Per: " + str(round(pca.fit(X).explained_variance_ratio_[0], 4)))
plt.ylabel("C2,Explained Var Per: " + str(round(pca.fit(X).explained_variance_ratio_[1], 4)))
plt.subplot(212)
plt.scatter(X_reduced2[:, 0], X_reduced2[:, 1], color=['red', 'blue'], alpha=.8, lw=2)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Dimension reduction with feature scaling')
plt.xlabel("C1,Explained Var Per: " + str(round(pca.fit(X_scaled).explained_variance_ratio_[0], 4)))
plt.ylabel("C2,Explained Var Per: " + str(round(pca.fit(X_scaled).explained_variance_ratio_[1], 4)))
plt.show()

# 从散点图来看,X_scaled的PCA降维把数据更好的区分了,但就这个数据而言并不适合PCA,第一是因为数据特征的维度不大,第二是我输出
# 了两个主成分的解释方差百分比,可以看到图二的解释方差百分比的和是很低的,不足以描述X数据的整体情况


# Q2.1
Z = linkage(X_scaled, method='complete', metric='euclidean')
Z1 = linkage(X_scaled, method='single', metric='euclidean')
plt.figure(figsize=(25, 10))
plt.subplot(211)
plt.title('Agglomerative clustering with complete linkage method')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8)

plt.subplot(212)
plt.title('Agglomerative clustering with single linkage method')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z1, leaf_rotation=90., leaf_font_size=8)

plt.show()
# Single-Link方法是选择两个不同聚类中距离最小的两个点,Complete-Link是最大距离的,所以产生了距离的差异


# Q2.2
Y = pdist(X_scaled, metric='euclidean')
Y1 = squareform(Y)
Y2 = VAT(X_scaled)

plt.figure(figsize=(25, 10))

plt.subplot(211)
sns.heatmap(Y1, annot=False, vmax=np.ceil(np.max(Y1)), square=True, cmap="Blues")
plt.title('Heatmap of Dissimilarity Matrix')
plt.xlabel('data1')
plt.ylabel('data2')

plt.subplot(212)
sns.heatmap(Y2[0], annot=False, vmax=np.ceil(np.max(Y2[0])), square=True, cmap="Blues")
plt.title('Heatmap of VAT')
plt.xlabel('data1')
plt.ylabel('data2')
plt.show()
# Complete-Link跟VAT是对应的,因为VAT最大距离就是Complete-Link中的最大距离,但VAT并没有给出很好的集群因为区分度并不是很明显...


# Q3.1
Y = np.corrcoef(X_scaled.T)

plt.subplot(111)

sns.heatmap(Y, annot=False, vmax=1, square=True, cmap="Blues", xticklabels=X_textx, yticklabels=X_textx, linewidths=1)
plt.title('Heatmap of Corr')

plt.show()
# pay_amt1-pay_amt6有着比较高的相关性,bill_amt1-bill_amt6有着较高的相关性,graduate_school和pay_amt3有着最低的相关性...

# Q3.2
num_list = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
            'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']
for i in num_list:
    x = pd.Series(df0[i])
    s = pd.cut(x, bins=[np.min(x), np.min(x) + (np.max(x) - np.min(x)) / 4, np.min(x) + (np.max(x) - np.min(x)) / 2,
                        np.min(x) + 3 * (np.max(x) - np.min(x)) / 4, np.max(x)])
    d = pd.get_dummies(s)
    print('######', i, '#######')
    print('bin  # 1: range [', np.min(x), ',', np.min(x) + (np.max(x) - np.min(x)) / 4, ')')
    print('bin  # 2: range [', np.min(x) + (np.max(x) - np.min(x)) / 4, ',', np.min(x) + (np.max(x) - np.min(x)) / 2,
          ')')
    print('bin  # 3: range [', np.min(x) + (np.max(x) - np.min(x)) / 2, ',',
          3 * np.min(x) + (np.max(x) - np.min(x)) / 4, ')')
    print('bin  # 4: range [', 3 * np.min(x) + (np.max(x) - np.min(x)) / 4, ',', np.max(x), ')')

mutual_infol = []
for i in X_textx:
    df1 = df0.copy()
    df1.index = range(len(df1['pay_2']))
    x = list(df1[i].values)
    Y = list(df1.loc[:, 'label'].values)
    mutual_infol.append(mutual_info(x, Y))

plt.figure()
ax1 = plt.subplot(111)
rect = ax1.bar(np.arange(len(X_textx)), mutual_infol, width=0.5, color="lightblue")
ax1.set_xticks(np.arange(len(X_textx)))
ax1.set_xticklabels(X_textx)
ax1.set_title("NMI")
ax1.grid(True)
ax1.set_ylim(0, 1)
plt.show()

# Q4.1
X = df0.loc[:, ['limit_bal', 'is_male', 'graduate_school', 'university', 'high_school', 'is_married', 'age',
                'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3',
                'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5',
                'pay_amt6']].values
Y = df0.loc[:, ['label']].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('X_train matrix: ', np.shape(X_train))
print('y_train labels: ', np.shape(y_train))
print('X_test matrix: ', np.shape(X_test))
print('y_test labels: ', np.shape(y_test))

# Q4.2
X_train = X_train.T
X_test = X_test.T

for i in range(np.shape(X_train)[0]):
    X_train[i] = (X_train[i] - np.mean(X_train[i])) / np.std(X_train[i])
    X_test[i] = (X_test[i] - np.mean(X_test[i])) / np.std(X_test[i])

X_train = X_train.T
X_test = X_test.T

# 不行,因为前面的X_scaled是对所有的X数据做的,把所有数据归一化,不能保留列与列(不同属性)数据的差异性

# Q4.3
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_train)
acc = round(accuracy_score(y_train, y_pred), 4) * 100
print('Train accuracy: ', acc, '%')

y_pred = neigh.predict(X_test)
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print('Test accuracy: ', acc, '%')

acc_all = []
time_all = []
for i in range(10):
    neigh = KNeighborsClassifier(n_neighbors=i + 1)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc_all.append(round(accuracy_score(y_test, y_pred), 4) * 100)
    time_all.append(i)

# 此处通过i=1-11的循环去获得ACC,然后展示最高ACC的值还有n_neighbors的取值
try:
    print('Best n_neighbors: ', time_all[np.where(acc_all == np.max(acc_all))[0][0]])
except:
    print('Best n_neighbors: ', time_all[np.where(acc_all == np.max(acc_all))[0]])
print('Best ACC: ', np.max(acc_all))

# Q4.4
mode = DecisionTreeClassifier()
mode.fit(X_train, y_train)
y_pred = mode.predict(X_train)
acc = round(accuracy_score(y_train, y_pred), 4) * 100
print('Train accuracy: ', acc, '%')

y_pred = mode.predict(X_test)
acc = round(accuracy_score(y_test, y_pred), 4) * 100
print('Test accuracy: ', acc, '%')

'''
不如KNN,符不符合预期怎么说随缘,可以参考这个网页
https://blog.csdn.net/u014563989/article/details/43797977
'''
