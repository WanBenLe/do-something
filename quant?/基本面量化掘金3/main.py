# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *
import pandas as pd
import numpy as np
from numba import jit
from analyst import runpostlasso
import warnings

warnings.filterwarnings("ignore")

'''
本策略每隔1个月定时触发计算SHSE.000300成份股的过去一天EV/EBITDA值并选取30只EV/EBITDA值最小且大于零的股票
对不在股票池的股票平仓并等权配置股票池的标的
并用相应的CFFEX.IF对应的真实合约等额对冲
回测数据为:SHSE.000300和他们的成份股和CFFEX.IF对应的真实合约
回测时间为:2017-07-01 08:00:00到2017-10-01 16:00:00
'''


@jit
def calcr(datax, params):
    d = params[0] + params[1:] * datax
    d = np.sum(d, axis=1)
    return d


def init(context):
    # 每月第一个交易日09:40:00的定时执行algo任务
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:40:00')

    # 设置开仓在股票和期货的资金百分比(期货在后面自动进行杠杆相关的调整)
    context.percentage_stock = 0.95
    context.dingshi = 0
    context.gogo = np.array([0, 0])
    context.trade_count=20


def algo(context):
    now = context.now.strftime("%Y-%m-%d")
    close = \
        history_n(symbol='SHSE.000016', frequency='1d', end_time=now, fields='close', count=1, adjust=1,
                  df=True)[
            'close'].values[0]
    temp = np.array([close, context.account().cash.nav])
    context.gogo = np.vstack((context.gogo, temp))

    if np.mod(context.dingshi, context.trade_count) == 0:
        print('时间', now)
        run_jump = 0
        context.dingshi = 0
        # 获取当前时刻

        lassox = runpostlasso(now,context.trade_count)
        if lassox.error_code == -1:
            print("本期没有变量显著,平所有仓位")
            run_jump = 1
            # fin = np.array([''])
        else:
            para_name = lassox.para_name
            params = lassox.params
            name_list = lassox.name_list

            # 获取上一个交易日
            last_day = get_previous_trading_date(exchange='SHSE', date=now)
            # 获取沪深300成份股
            stock300 = get_history_constituents(index='SHSE.000016', start_date=last_day,
                                                end_date=last_day)[0]['constituents'].keys()

            # 获取当天有交易的股票
            not_suspended_info = get_history_instruments(symbols=stock300, start_date=now, end_date=now)
            not_suspended_symbols = [item['symbol'] for item in not_suspended_info if not item['is_suspended']]
            # 获取成份股EV/EBITDA大于0并为最小的30个
            fin = get_fundamentals(table='trading_derivative_indicator', symbols=not_suspended_symbols,
                                   start_date=now, end_date=now, fields=para_name, df=True)
            datax = fin[name_list].values

            fin['forcast'] = calcr(datax, params)
            fin = fin[['symbol', 'forcast']]

            fin = fin.sort_values(by='forcast', ascending=False).values
            if np.max(fin[:, 1]) < 0:
                print('预测收益率最大收益率`小于0,平所有仓位')
                # fin = np.array([''])
                fin = fin[0:20, 0]
            else:
                fin = fin[0:20, 0]

        if run_jump != 1:
            # 获取当前仓位
            positions = context.account().positions()
            # 平不在标的池或不为当前股指期货主力合约对应真实合约的标的
            for position in positions:
                symbol = position['symbol']
                if symbol not in fin:
                    order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
                                         position_side=PositionSide_Long)
                    print('市价单平不在标的池的', symbol)

            # 获取股票的权重
            percent = context.percentage_stock / len(fin)
            # 买在标的池中的股票
            for symbol in fin:
                order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
                                     position_side=PositionSide_Long)
                # print(symbol, '以市价单调多仓到仓位', percent)

    context.dingshi += 1


def on_backtest_finished(context, indicator):
    print(indicator)
    pd.DataFrame(context.gogo, columns=['X50', 'NAV']).to_csv('nav.csv', index=False)


if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='93dab158-e45d-11ea-bd00-309c23ff624b',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='b81189eb81a34b3fe92e3ee339430f9fafb87d68',
        backtest_start_time='2019-08-20 08:00:00',
        backtest_end_time='2020-08-20 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.0025)
