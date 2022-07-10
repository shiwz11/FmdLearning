---
title: QUANTAXIS-savedata
author: Shiwz11
date: '2022-07-10'
slug: quantaxis-savedata
categories:
  - Memos
tags:
  - QUANT
---

如何向QUANTAXIS-mongodb存储自己的数据

今天摸索了下怎么把利率债的数据存储到QUANTAXIS的mongodb里。数据源是同花顺iFind。同花顺的python数据接口那点数据量根本不够用。。。

根据QA的`QA_SU_save_future_day`等相关函数改写而来。如果本文内容涉及侵权，请联系作者`shiwz11@hotmail.com`会立刻做相关处理。

大致分为两部分：

- 存储bond_list

- 存储bond_day即bond的日线数据

# 导入必要的包和函数

```{python}
import pymongo
import datetime
import pandas as pd
import numpy as np
import QUANTAXIS as QA
from QUANTAXIS.QAUtil import (
#    DATABASE,
    QA_util_to_json_from_pandas,
    QA_util_log_info,
    QA_util_code_tolist,
    QA_util_get_next_day,
    QA_util_date_valid,
    QA_util_date_stamp,
)

from QUANTAXIS.QAData.data_resample import (
    QA_data_futureday_resample
)

from QUANTAXIS.QAUtil.QASetting import QA_Setting
from QUANTAXIS.QASU.save_tdx import now_time
from QUANTAXIS.QAData.base_datastruct import _quotation_base
from functools import lru_cache, partial, reduce
from iFinDPy import *
QASETTING = QA_Setting()
DATABASE = QASETTING.client.quantaxis
# ifind username and password
THS_iFinDLogin("xxxx", "xxxxxx")
```

# 定义数据结构

为与QUANTAXIS的数据结构保持一致，改写现有代码定义了`QA_DataStruct_Bond_day_shi`类。

```{python}
# QA_DataStruct_Bond_day 类
class QA_DataStruct_Bond_day_shi(_quotation_base):

    def __init__(self, DataFrame, dtype='bond_day', if_fq=''):
        super().__init__(DataFrame, dtype, if_fq)
        self.type = 'bond_day'
        self.data = self.data.loc[:,
                                  [
                                    'ths_evaluate_yield_cb_bond',
                                    'ths_valuate_full_price_cb_bond',
                                    'ths_evaluate_modified_dur_cb_bond',
                                    'ths_open_price_bond',
                                    'ths_close_daily_settle_bond',
                                    'ths_high_price_bond',
                                    'ths_low_bond',
                                    'ths_avg_price_bond', 
                                    'ths_vol_bond', 
                                    'ths_trans_amt_bond',
                                  ]]
        self.if_fq = if_fq

    # 抽象类继承
    def choose_db(self):
        self.mongo_coll = DATABASE.bond_day_shi

    def __repr__(self):
        return '< QA_DataStruct_Bond_day_shi with {} securities >'.format(
            len(self.code)
        )

    __str__ = __repr__

    @property
    @lru_cache()
    def week(self):
        return self.resample('w')

    @property
    @lru_cache()
    def month(self):
        return self.resample('M')

    @property
    @lru_cache()
    def quarter(self):
        return self.resample('Q')

    @property
    @lru_cache()
    def tradedate(self):
        """返回交易所日历下的日期

        Returns:
            [type] -- [description]
        """

        try:
            return self.date
        except:
            return None

    @property
    @lru_cache()
    def tradetime(self):
        """返回交易所日历下的日期

        Returns:
            [type] -- [description]
        """

        try:
            return self.date
        except:
            return None

    # @property
    # @lru_cache()
    # def semiannual(self):
    #     return self.resample('SA')

    @property
    @lru_cache()
    def year(self):
        return self.resample('Y')

    def resample(self, level):
        try:
            return self.add_func(QA_data_futureday_resample, level).sort_index()
        except Exception as e:
            print('QA ERROR : FAIL TO RESAMPLE {}'.format(e))
            return None

```

# 存储和读取bond_list

## 存储bond_list

`bond_list`数据是从ifind手动下载的，保存为excel之后，再读取然后存入mongodb。

```{python}
# 存储bond_list
def Shi_read_bond_list_local():
    """
    这里的category=9.0是随意设的
    market=99，暂时代银行间市场

    Args:
        path (_type_, optional): _description_. Defaults to local_data_excel_path.
        file_name (str, optional): _description_. Defaults to "bond_list.xlsx".
    """
    bond_list = pd.read_excel(r"C:\Users\shiwz11\Documents\dataswap\bond_list.xlsx",index_col="code")
    bond_list['category'], bond_list["market"], bond_list["desc"], bond_list["value"] = [9.0, 99.0, "", np.nan]
    bond_list = bond_list.reset_index().set_index(["code"], drop=False)
    return bond_list


def QA_SU_save_bond_list_shi(client=DATABASE, ui_log=None, ui_progress=None):
    """
    保存债券代码列表
    Args:
        client (_type_, optional): _description_. Defaults to DATABASE.
        ui_log (_type_, optional): _description_. Defaults to None.
        ui_progress (_type_, optional): _description_. Defaults to None.
    """
    bond_list = Shi_read_bond_list_local()
    coll_bond_list = client["bond_list_shi"]
    coll_bond_list.create_index("code", unique=True)
    try:
        coll_bond_list.insert_many(
            QA_util_to_json_from_pandas(bond_list),
            ordered=False
        )
    except:
        pass
```

存储的时候，调用此函数：

```{python}
QA_SU_save_bond_list_shi()
```

## 读取bond_list

存储成功之后就是读取了，为了保持和QA的函数及数据结构一致，改写了以下函数：

```{python}
# 读取bond_list
def QA_fetch_bond_list_shi(collections=DATABASE.bond_list_shi):
    '获取债券列表'
    return pd.DataFrame([item for item in collections.find()]).drop(
        '_id',
        axis=1,
        inplace=False
    ).set_index(
        'code',
        drop=False
    )
```

## 存储利率债数据

这一部分，直接调用ifind的接口然后循环获取数据之后，存入mongodb，保持了和QA函数的一致。

```{python}
# 存储债券数据
def QA_fetch_get_bond_day_shi(code, start_date, end_date):
    try:
        bond_day = THS_DS(code,'ths_evaluate_yield_cb_bond;ths_valuate_full_price_cb_bond;ths_evaluate_modified_dur_cb_bond;ths_open_price_bond;ths_close_daily_settle_bond;ths_high_price_bond;ths_low_bond;ths_avg_price_bond;ths_vol_bond;ths_trans_amt_bond','100;100;100;103;103;103;103;103;;','Fill:Blank',start_date,end_date)
        bond_day_data = bond_day.data
        bond_day_data.rename(columns={"time":"date", "thscode":"code"}, inplace=True)
        bond_day_data.reset_index().set_index(["date", "code"], inplace=True)
        bond_day_data["date_stamp"] = bond_day_data.date.apply(lambda x: QA_util_date_stamp(x))
        if bond_day.data.shape[0] == 0:
            raise Exception("数据为空")

    except:
       raise Exception("Ukown:同花顺接口调用错误！")
    return bond_day_data

def get_start_end_date(code):
    try:
        # 同花顺接口免费版只提供近5年数据
        ths_limit_date = pd.Timestamp(now_time())-pd.Timedelta(days = 360*5)
        timedf = THS_BD(code,'ths_listed_date_bond;ths_stop_listing_date_bond',';').data
        start_date = pd.Timestamp(timedf.iat[0,1])
        end_date = pd.Timestamp(timedf.iat[0,2])
        now_date = pd.Timestamp(now_time())
        if start_date >= now_date:
            raise Exception('上市开始日期不对！')
        elif start_date <= ths_limit_date:
            start_date = ths_limit_date
        elif end_date <= now_date:
            end_date = end_date
        else:
            end_date = now_date
    except:
        raise Exception("Ukown:同花顺接口调用错误！")
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def QA_SU_save_bond_day_shi(client=DATABASE, ui_log=None, ui_progress=None):
    '''
    save bond_day
    保存日线数据
    :param client:
    :param ui_log:  给GUI qt 界面使用
    :param ui_progress: 给GUI qt 界面使用
    :param ui_progress_int_value: 给GUI qt 界面使用
    :return:
    '''
    bond_list = [
        item for item in QA_fetch_bond_list_shi().code.unique().tolist()
    ]
    coll_bond_day = client["bond_day_shi"]
    coll_bond_day.create_index(
        [("code",
          pymongo.ASCENDING),
         ("date_stamp",
          pymongo.ASCENDING)]
    )
    err = []

    def __saving_work(code, coll_bond_day):
        try:
            QA_util_log_info(
                '##JOB12 Now Saving Bond_DAY==== {}'.format(str(code)),
                ui_log
            )

            # 首选查找数据库 是否 有 这个代码的数据
            ref = coll_bond_day.find({'code': str(code)})
            end_date = str(now_time())[0:10]

            # 当前数据库已经包含了这个代码的数据， 继续增量更新
            # 加入这个判断的原因是因为如果债券是刚上市的 数据库会没有数据 所以会有负索引问题出现
            if ref.count() > 0:

                # 接着上次获取的日期继续更新
                start_date = ref[ref.count() - 1]['date']

                QA_util_log_info(
                    'UPDATE_Bond_DAY_shi \n Trying updating {} from {} to {}'
                        .format(code,
                                start_date,
                                end_date),
                    ui_log
                )
                if start_date != end_date:
                    coll_bond_day.insert_many(
                        QA_util_to_json_from_pandas(
                            QA_fetch_get_bond_day_shi(
                                str(code),
                                QA_util_get_next_day(start_date),
                                end_date
                            )
                        )
                    )

            # 当前数据库中没有这个代码的股票数据， 从start_date 开始下载所有的数据
            else:
                start_date, end_date = get_start_end_date(code=code)
                QA_util_log_info(
                    'UPDATE_Bond_DAY \n Trying updating {} from {} to {}'
                        .format(code,
                                start_date,
                                end_date),
                    ui_log
                )
                if start_date != end_date:
                    coll_bond_day.insert_many(
                        QA_util_to_json_from_pandas(
                            QA_fetch_get_bond_day_shi(
                                str(code),
                                start_date,
                                end_date
                            )
                        )
                    )
        except Exception as error0:
            print(error0)
            err.append(str(code))
            
    for item in range(len(bond_list)):
        QA_util_log_info('The {} of Total {}'.format(item, len(bond_list)))

        strProgressToLog = 'DOWNLOAD PROGRESS {} {}'.format(
            str(float(item / len(bond_list) * 100))[0:4] + '%',
            ui_log
        )
        intProgressToLog = int(float(item / len(bond_list) * 100))
        QA_util_log_info(
            strProgressToLog,
            ui_log=ui_log,
            ui_progress=ui_progress,
            ui_progress_int_value=intProgressToLog
        )

        __saving_work(bond_list[item], coll_bond_day)

    if len(err) < 1:
        QA_util_log_info('SUCCESS save bond day ^_^', ui_log)
    else:
        QA_util_log_info(' ERROR CODE \n ', ui_log)
        QA_util_log_info(err, ui_log)    

def QA_SU_save_single_bond_day_shi(code : str, client=DATABASE, ui_log=None, ui_progress=None):
    '''
    save single_bond_day
    保存单个债券数据日线数据
    :param client:
    :param ui_log:  给GUI qt 界面使用
    :param ui_progress: 给GUI qt 界面使用
    :param ui_progress_int_value: 给GUI qt 界面使用
    :return:
    '''
    coll_bond_day = client["bond_day_shi"]
    coll_bond_day.create_index(
        [("code",
          pymongo.ASCENDING),
         ("date_stamp",
          pymongo.ASCENDING)]
    )
    err = []

    def __saving_work(code, coll_bond_day):
        try:
            QA_util_log_info(
                '##JOB12 Now Saving Bond_DAY==== {}'.format(str(code)),
                ui_log
            )

            # 首选查找数据库 是否 有 这个代码的数据
            ref = coll_bond_day.find({'code': str(code)})
            end_date = str(now_time())[0:10]

            # 当前数据库已经包含了这个代码的数据， 继续增量更新
            # 加入这个判断的原因是因为如果债券是刚上市的 数据库会没有数据 所以会有负索引问题出现
            if ref.count() > 0:

                # 接着上次获取的日期继续更新
                start_date = ref[ref.count() - 1]['date']

                QA_util_log_info(
                    'UPDATE_Bond_DAY \n Trying updating {} from {} to {}'
                        .format(code,
                                start_date,
                                end_date),
                    ui_log
                )
                if start_date != end_date:
                    coll_bond_day.insert_many(
                        QA_util_to_json_from_pandas(
                            QA_fetch_get_bond_day_shi(
                                str(code),
                                QA_util_get_next_day(start_date),
                                end_date
                            )
                        )
                    )

            # 当前数据库中没有这个代码的股票数据， 从1990-01-01 开始下载所有的数据
            else:
                start_date, end_date = get_start_end_date(code=code)
                QA_util_log_info(
                    'UPDATE_Bond_DAY \n Trying updating {} from {} to {}'
                        .format(code,
                                start_date,
                                end_date),
                    ui_log
                )
                if start_date != end_date:
                    coll_bond_day.insert_many(
                        QA_util_to_json_from_pandas(
                            QA_fetch_get_bond_day_shi(
                                str(code),
                                start_date,
                                end_date
                            )
                        )
                    )
        except Exception as error0:
            print(error0)
            err.append(str(code))


    __saving_work(code, coll_bond_day)

    if len(err) < 1:
        QA_util_log_info('SUCCESS save Bond day ^_^', ui_log)
    else:
        QA_util_log_info(' ERROR CODE \n ', ui_log)
        QA_util_log_info(err, ui_log)
```

## 读取bond的日线数据

这部分用到了一开始定义的data struct。

```{python}
# 读取bond数据
def QA_fetch_bond_day_shi_adv(
    code,
    start,
    end=None,
    if_drop_index=True,
                                   # 🛠 todo collections 参数没有用到， 且数据库是固定的， 这个变量后期去掉
    collections=DATABASE.bond_day_shi
):
    '''
    :param code: code:  字符串str eg 600085
    :param start:  字符串str 开始日期 eg 2011-01-01
    :param end:  字符串str 结束日期 eg 2011-05-01
    :param if_drop_index: Ture False ， dataframe drop index or not
    :param collections:  mongodb 数据库
    :return:
    '''
    '获取债券日线'
    end = start if end is None else end
    start = str(start)[0:10]
    end = str(end)[0:10]

    # 🛠 todo 报告错误 如果开始时间 在 结束时间之后
    # 🛠 todo 如果相等

    res = QA_fetch_bond_day_shi(code, start, end, format='pd', collections= collections)
    if res is None:
        print(
            "QA Error QA_fetch_future_day_adv parameter code=%s start=%s end=%s call QA_fetch_future_day return None"
            % (code,
               start,
               end)
        )
    else:
        res_set_index = res.set_index(['date', 'code'])
        # if res_set_index is None:
        #     print("QA Error QA_fetch_index_day_adv set index 'date, code' return None")
        #     return None
        return QA_DataStruct_Bond_day_shi(res_set_index)


def QA_fetch_bond_day_shi(
    code,
    start,
    end,
    format='numpy',
    collections=DATABASE.bond_day_shi
):
    start = str(start)[0:10]
    end = str(end)[0:10]
    code = QA_util_code_tolist(code, auto_fill=False)

    if QA_util_date_valid(end) == True:

        _data = []
        cursor = collections.find(
            {
                'code': {
                    '$in': code
                },
                "date_stamp":
                    {
                        "$lte": QA_util_date_stamp(end),
                        "$gte": QA_util_date_stamp(start)
                    }
            },
            {"_id": 0},
            batch_size=10000
        )
        if format in ['dict', 'json']:
            return [data for data in cursor]
        for item in cursor:

            _data.append(
                [
                    str(item['code']),
                    float(item['ths_evaluate_yield_cb_bond']),
                    float(item['ths_valuate_full_price_cb_bond']),
                    float(item['ths_evaluate_modified_dur_cb_bond']),
                    float(item['ths_open_price_bond']),
                    float(item['ths_close_daily_settle_bond']),
                    float(item['ths_high_price_bond']),
                    float(item['ths_low_bond']),
                    float(item['ths_avg_price_bond']), 
                    float(item['ths_vol_bond']), 
                    float(item['ths_trans_amt_bond']),
                    item['date']
                ]
            )

        # 多种数据格式
        if format in ['n', 'N', 'numpy']:
            _data = np.asarray(_data)
        elif format in ['list', 'l', 'L']:
            _data = _data
        elif format in ['P', 'p', 'pandas', 'pd']:
            _data = pd.DataFrame(
                _data,
                columns=[
                        'code',
                        'ths_evaluate_yield_cb_bond',
                        'ths_valuate_full_price_cb_bond',
                        'ths_evaluate_modified_dur_cb_bond',
                        'ths_open_price_bond',
                        'ths_close_daily_settle_bond',
                        'ths_high_price_bond',
                        'ths_low_bond',
                        'ths_avg_price_bond', 
                        'ths_vol_bond', 
                        'ths_trans_amt_bond',
                        'date'
                ]
            ).drop_duplicates()
            _data['date'] = pd.to_datetime(_data['date'], utc=False)
            _data = _data.set_index('date', drop=False)
        else:
            print(
                "QA Error QA_fetch_future_day format parameter %s is none of  \"P, p, pandas, pd , n, N, numpy !\" "
                % format
            )
        return _data
    else:
        QA_util_log_info('QA something wrong with date')


```

读取时，使用以下代码：

```{python}
df210215 = QA_fetch_bond_day_shi_adv(code = "210215.IB", start="2022-06-01", end = "2022-07-01")
```

到此结束！