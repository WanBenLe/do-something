# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:49:52 2017

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout, regularizers
from keras.models import Sequential
from keras.utils import np_utils

dfos = 'E:\\0AAA\\0train.csv'
train = pd.read_csv(open(dfos))
del train['Unnamed: 0']
del train['ldate']
outputfile = 'output.xls'
modelfile = 'modelweight.model'
feature = ['ma3', 'PER', 'PBR', 'PSR', 'EPS', 'bvps', 'cfps', 'afps']
print(len(train['rank']))
# 因子
label = ['rank']
x_train = train[feature].as_matrix()
y_train = train[label].as_matrix()
y_train = np_utils.to_categorical(y_train, 5)
model = Sequential()
model.add(Dense(output_dim=(240 * 20), input_dim=8, activation='relu', use_bias=True,
                kernel_regularizer=regularizers.l2(0.01), kernel_initializer='RandomNormal'))  # 输入
Dropout(0.2)
model.add(Dense(output_dim=5, input_dim=(240 * 20), activation='softmax'))  # 输出
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # MSE.ADAM
hist = model.fit(x_train, y_train, nb_epoch=50, batch_size=500, validation_split=0.01, shuffle=True)
print(hist.history)
model.save_weights(modelfile)
pltx = np.linspace(0, 50, 50, endpoint=True)
pltc1 = hist.history['acc']
pltc2 = hist.history['val_acc']
plt.plot(pltx, pltc1)
plt.plot(pltx, pltc2, color='black')
plt.show
