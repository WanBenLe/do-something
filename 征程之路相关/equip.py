import pandas as pd
import numpy as np
import re

id_text = open('equip.txt', 'r', encoding='UTF-8')
all_text = id_text.readlines()

textfree = np.chararray((1, 16), itemsize=255, unicode=1)
textfree[0] = ['力量增幅', '敏捷增幅', '智力增幅', '爆伤概率', '生命增幅', '移动速度', '伤害减免', '冷却缩减', '技能增幅', '爆伤伤害', '法术吸血', '攻击增幅',
               '防御增幅', '魔法抗性', '全属性增幅', '基础攻击间隔']
len_shuxing=len(textfree[0])
len_all=len_shuxing+3


ben_go = np.chararray((1, len_all), itemsize=255, unicode=1)
ben_go[0] = ['名称', '品质', '描述', '力量增幅', '敏捷增幅', '智力增幅', '爆伤概率', '生命增幅', '移动速度', '伤害减免', '冷却缩减', '技能增幅', '爆伤伤害',
             '法术吸血', '攻击增幅', '防御增幅', '魔法抗性', '全属性增幅', '基础攻击间隔']
print(all_text)
new_equip = 1
new_sta = 0
for ii in range(len(all_text)):
    if new_equip == 1:
        new_dec = np.chararray((1, len_all), itemsize=255, unicode=1)
        if all_text[ii][0:3] == '<b>':
            new_dec[0, 0] = all_text[ii][3:-1]
            new_equip = 0
    elif all_text[ii][0:3] == '品质:':
        new_dec[0, 1] = all_text[ii][3:-1]
    elif all_text[ii][0:3] == '属性:':
        new_sta = 1
        new_row = 1
    elif (all_text[ii] != '\n') and (new_sta == 1):
        if new_row != 1:
            new_dec[0, 2] = np.char.add(new_dec[0, 2], '\r\n-  ')
        new_row = 0
        shuxing_new = all_text[ii][:-1]
        # 描述词条
        new_dec[0, 2] = np.char.add(new_dec[0, 2], shuxing_new)
        # 正则表达式匹配获取第一个整数
        number1 = re.findall(r'-?[1-9]\d*', shuxing_new)

        if len(number1) > 0:
            number = number1[0]
            for jj in range(len_shuxing):

                # 检定不为魔法抗性或基础攻击间隔或全属性的情况
                if (jj != 14) and (jj != 13) and (jj != 15):
                    if shuxing_new[0:2] == '增加':
                        # 匹配属性
                        if shuxing_new[-4:] == textfree[0, jj]:
                            new_dec[0, 3 + jj] = number
                            break
                elif jj == 13:
                    if shuxing_new[1:5] == '魔法抗性':
                        new_dec[0, 3 + jj] = number
                        break

                elif jj == 14:
                    if shuxing_new[0:2] == '增加':
                        # 匹配全属性增幅
                        if shuxing_new[-5:] == textfree[0, jj]:
                            new_dec[0, 3 + jj] = number
                            break

                elif jj == 15:
                    if shuxing_new[0:2] == '降低':
                        # 匹配基础攻击间隔
                        if shuxing_new[-6:] == textfree[0, jj]:
                            new_dec[0, 3 + jj] = re.findall(r'[-+]?(\b[0-9]+(\.[0-9]+)?|\.[0-9]+)', shuxing_new)[0][0]
                            break

    elif all_text[ii] == '\n':
        new_equip = 1
        new_sta = 0
        ben_go = np.vstack((ben_go, new_dec))
ben_go = np.vstack((ben_go, new_dec))
ben = np.chararray((1, len_all), itemsize=255, unicode=1)
ben[0, 0:3] = ['西西弗斯的萌新', '汪~汪~汪!', 'dalao please daidaiwo~']
ben_go = np.vstack((ben_go, ben))
ben = np.chararray((1, len_all), itemsize=255, unicode=1)
ben[0, 0:3] = ['本物品描述', '专属-征程之路宁静9群', '欢迎来9群(835821812)玩耍~']
ben_go = np.vstack((ben_go, ben))
well = pd.DataFrame(ben_go)
well.to_csv('装备描述-Ben.csv', encoding='ansi', index=0)
print('finish')
