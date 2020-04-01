import tushare as ts

a = ts.get_h_data('000001', index=True, start='2014-01-01', end='2018-06-01')
b = ts.get_h_data('000034', start='2014-01-01', end='2018-06-01')
a['34'] = b['close']

print(a)
# 选择保存
a.to_csv('C:/Users/Administrator/Desktop/biye/000875.csv')
