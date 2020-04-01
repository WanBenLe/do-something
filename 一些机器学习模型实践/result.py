import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree

# read data
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
print(data1)
print(data2)

# 'null' insted
data1x = data1.values
data2x = np.transpose(data2.values)[0]
a = 0
b = 0
for i in range(len(data2x)):
    if data2x[i] == 'null':
        data2x[i] = np.nan
    else:
        data2x[i] = int(data2x[i])
        a += 1
        b += data2x[i]

# mean expect nan
nanmeanx = int(b / a)
print(data2x.reshape(-1, 1))

# nanmean instead nan
for i in range(len(data2x)):
    if np.isnan(data2x[i]):
        data2x[i] = nanmeanx
data2x = data2x.reshape(-1, 1)
print(data2x)

# boxplot
t = np.transpose(data1x)

for i in range(len(t)):
    plt.boxplot(t[i], sym='o', whis=0.05)
    plt.show()
plt.boxplot(data2x, sym='o', whis=0.05)
plt.show()

# outlier
for i in range(len(t)):
    t1 = np.mean(t[i]) + 3 * np.std(t[i])
    t2 = np.mean(t[i]) - 3 * np.std(t[i])
    for j in range(len(t[0])):
        if t[i][j] > t1:
            t[i][j] = t1
        elif t[i][j] < t2:
            t[i][j] = t2

t1 = np.ceil(np.mean(data2x) + 3 * np.std(data2x))
t2 = np.floor(np.mean(data2x) - 3 * np.std(data2x))
for j in range(len(data2x)):
    if data2x[j] > t1:
        data2x[j] = t1
    elif data2x[j] < t2:
        data2x[j] = t2

data1x = np.transpose(t)
print(data1x)
print(data2x)

# scatter-plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('self-inflicted ')
plt.ylabel('Person Number')
ax1.scatter(range(80), data2x, c='r', marker='o')
plt.show()

# Corr
data = np.hstack((data1x, data2x))
Y = np.corrcoef(data.astype(np.float64).T)
plt.subplot(111)
sns.heatmap(Y, annot=False, vmax=1, square=True, cmap="Blues", linewidths=1)
plt.title('Heatmap of Corrcoef')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data1x, data2x, test_size=0.2, random_state=42)

clf = tree.DecisionTreeRegressor()
clf.fit(X_train, y_train)

forcast = clf.predict(X_test)
allx = 0.0
ally = 0.0
for i in range(len(forcast)):
    allx += (forcast[i] - np.mean(y_test)) ** 2
    ally += (y_test[i] - np.mean(y_test)) ** 2
print(allx/ally)

