#!/usr/bin/python3
# -*- coding:utf8 -*-
from math import *

file_obj = open("EXPERIENCEGATE.DAT.TXT", encoding="utf-8")
all_lines = file_obj.readlines()

t = 1.04


def omg(woowa, ta):
    a = woowa
    b = a.rfind('e')
    if b != -1:
        c = a[b + 2:]
        d = float(a[:b - 1]) * 10 ** int(c)
    else:
        d = float(a)
    d = floor(d / ta)
    return d


for line in all_lines:
    # 每行循环
    k = line.rfind('<FLOAT>Y:')
    if k != -1:
        woow = line[k + 9:]
        news = omg(woow, t)
        xy = line[:k + 9] + str(news)
    else:
        xy = line
    newfile = open('helyo.txt', 'a', encoding='utf-8')
    newfile.write(xy + '\n')
    newfile.close()
file_obj.close()
print('ok')
