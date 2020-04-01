from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import lightgbm as lgb

data0 = pd.read_csv('data.csv', encoding='utf-8')
data0.sample(frac=1)
data1 = data0.copy()
len_all = int(len(data0['逾期次数']) * 0.7)
class_all = data0.iloc[:, 15:].values
num_all = data0.iloc[:, 0:15].values

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(class_all)
data01 = enc.transform(class_all).toarray()

num_all = num_all.T

for i in range(len(num_all)):
    num_all[i][np.isnan(num_all[i])] = np.nanmean(num_all[i])
    num_all[i] = (num_all[i] - np.mean(num_all[i])) / np.std(num_all[i])

data0 = np.hstack((num_all.T, data01))

y_all = data0.T[0].T
x_all = data0.T[1:].T

y_tr = y_all[0:len_all]
y_te = y_all[len_all:]
x_tr = x_all[0:len_all]
x_te = x_all[len_all:]
print('ready')

train_data = lgb.Dataset(x_tr, label=y_tr)
test_data = lgb.Dataset(x_te, label=y_te)
num_round = 10
param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'regression_l2', 'metric': 'l2'}
# lgb.cv(param, train_data, num_round, nfold=5)
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
bst.save_model('model.txt')

bst = lgb.Booster(model_file='model.txt')
ypred = bst.predict(x_all)
ypred += np.min(ypred)
ypred = 100 - (ypred - np.min(ypred)) / (np.max(ypred) - np.min(ypred)) * 100
kmeans = KMeans(n_clusters=3, random_state=0).fit(ypred.reshape(-1, 1))
woow = kmeans.predict(ypred.reshape(-1, 1))

dcore = np.hstack((woow.reshape(-1, 1), ypred.reshape(-1, 1), data1.values.reshape(-1, 35)))
df_r = pd.DataFrame(dcore)
df_r.to_csv('score.csv', index=False, encoding='utf_8_sig')
print('finish')
'''
def get_Batch(data, label, batch_size):
    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32,
                                      allow_smaller_final_batch=False)
    return x_batch, y_batch


x_batch, y_batch = get_Batch(x_tr, y_tr, 1000)

x = tf.placeholder(tf.float32, [None, 191])
y = tf.placeholder(tf.float32, [None, 1])

weights_L1 = tf.Variable(tf.zeros([191, 95]))
biases_L1 = tf.Variable(tf.random_normal([95]))
wx_plus_b_L1 = tf.matmul(x, weights_L1) + biases_L1
L1 = tf.nn.relu(wx_plus_b_L1)

weights_L2 = tf.Variable(tf.random_normal([95, 38]))
biases_L2 = tf.Variable(tf.random_normal([38]))
wx_plus_b_L2 = tf.matmul(L1, weights_L2) + biases_L2
L2 = tf.nn.relu(wx_plus_b_L2)

weights_L3 = tf.Variable(tf.zeros([38, 1]))
biases_L3 = tf.Variable(tf.zeros([1]))
wx_plus_b_L3 = tf.matmul(L2, weights_L3) + biases_L3

predictions = tf.nn.sigmoid(wx_plus_b_L3)

MSE = tf.reduce_mean(tf.square(predictions - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(MSE)

train_loss = []
test_loss = []
test_accarucy = []
epoch_x = 51


def Accuracy(test_result, test_label):
    predict_ans = []
    label = []
    for (test, _label) in zip(test_result, test_label):
        test = np.exp(test)
        test = test / np.sum(test)
        predict_ans.append(np.argmax(test))
        label.append(np.argmax(_label))
    return accuracy_score(label, predict_ans)


print('run')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_x):
        for batch in range(int(len(y_tr) / 1000)):
            data, label = sess.run([x_batch, y_batch])
            sess.run(train_step, feed_dict={x: data, y: label})
        _train_loss = sess.run(MSE, feed_dict={x: x_tr, y: y_tr})
        print('train_loss', _train_loss)
        train_loss.append(_train_loss)
        # 测试损失
        _test_loss = sess.run(MSE, feed_dict={x: x_te, y: y_te})
        test_loss.append(_test_loss)
        print('test_loss', _test_loss)
        test_result = sess.run(predictions, feed_dict={x: x_te})
        test_accarucy.append(Accuracy(test_result, y_te))

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 结果可视化
col = ['Train_Loss', 'Test_Loss']
epoch = np.arange(epoch_x)
plt.plot(epoch, train_loss, 'r')
plt.plot(epoch, test_loss, 'b-.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(labels=col, loc='best')
plt.savefig('./训练与测试损失.jpg')
plt.show()
plt.close()

plt.plot(epoch, test_accarucy, 'r')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('./测试精度.jpg')
plt.show()
plt.close()
'''
