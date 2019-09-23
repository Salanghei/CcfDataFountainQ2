#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: zhaoyang
@license: (C) Copyright 2001-2019 Python Software Foundation. All rights reserved.
@contact: 1805453683@qq.com
@file: preprocess.py
@time: 2019/9/20 12:47
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 合并原数据中的年、月
def parse_date(x):
    return datetime.strptime(x, "%Y %m")

# 设置字体，使图中可以显示中文
plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["font.serif"] = ["KaiTi"]

######## 加载数据 ########
# 历史销量数据
sales_set = pd.read_csv("../data/train_sales_data.csv")
sales_set["date"] = sales_set["regYear"].map(str) + "-" + sales_set["regMonth"].map(str)
sales_set["date"] = pd.to_datetime(sales_set["date"])
sales_set["date"] = [datetime.strftime(x, "%Y-%m") for x in sales_set["date"]]  # 统一date的格式
# 车型搜索数据
search_set = pd.read_csv("../data/train_search_data.csv", parse_dates=[["regYear", "regMonth"]], date_parser=parse_date)
search_set.columns = ["date", "province", "adcode", "model", "popularity"]
search_set["date"] = [datetime.strftime(x, "%Y-%m") for x in search_set["date"]]  # 统一date的格式
# 汽车垂直媒体新闻评论数据和车型评论数据
comment_set = pd.read_csv("../data/train_user_reply_data.csv", parse_dates=[["regYear", "regMonth"]], date_parser=parse_date)
comment_set.columns = ["date", "model", "newsReplyVolume", "carCommentVolume"]
comment_set["date"] = [datetime.strftime(x, "%Y-%m") for x in comment_set["date"]]  # 统一date的格式
# 按照时间合并数据表
data_set = pd.merge(sales_set, search_set, how="left")
data_set = pd.merge(data_set, comment_set, how="left")
data_set.dropna()             # 删除有空值的行
data_set.drop_duplicates()    # 删除有重复的行

######## 绘制汽车销量和相关媒体信息随月份的变化趋势 ########
column_list = ["salesVolume", "popularity", "newsReplyVolume", "carCommentVolume"]
plt.figure(figsize=(8, 6))
for column_i in range(0, len(column_list)):
    temp_data = data_set[["date", column_list[column_i]]].groupby("date").agg("sum")
    plt.subplot(len(column_list), 1, column_i + 1)
    plt.title(column_list[column_i], y=0.5, loc="right")
    if column_i == len(column_list) - 1:
        plt.xticks(rotation=90)
    else:
        plt.xticks([])
    plt.plot(temp_data)
plt.show()

######## 绘制各省的汽车销量随月份的变化 ########
province_sales_data_list = data_set.groupby("province")
province_list = []
plt.figure(figsize=(8, 6))
for province, province_sales_data in province_sales_data_list:
    province_mon_sales_data = province_sales_data[["date", "salesVolume"]].groupby("date").agg("sum")
    plt.xticks(rotation=90)
    plt.plot(province_mon_sales_data, label=province)
    province_list.append(province)
plt.legend(province_list)
plt.show()

######## 计算到最终预测日期相隔的月份 ########
date_list = data_set.groupby("date").size().index
for date in date_list:
    years = 2018 - int(date_list[0].split("-")[0])
    mons = years * 12 + (4 - int(date_list[0].split("-")[1]))
    data_set.replace(date, mons, inplace=True)

######## 处理离散值 ########
# 将车型信息替换为0-59的数字
model_list = data_set.groupby("model").size().index
index = 0
for model in model_list:
    data_set.replace(model, index, inplace=True)
    index += 1

# # 将车身信息替换为0-3的数字
# bodyType_list = data_set.groupby("bodyType").size().index
# index = 0
# for bodyType in bodyType_list:
#     data_set.replace(bodyType, index, inplace=True)
#     index += 1

######## 处理离散值：独热编码 ########
province_map = {
    "上海": "sh", "云南": "yn", "内蒙古": "nmg", "北京": "bj", "四川": "sc", "安徽": "ah", "山东": "sd", "山西": "sx1",
    "广东": "gd", "广西": "gx", "江苏": "js", "江西": "jx", "河北": "hb", "河南": "hn", "浙江": "zj", "湖北": "hb",
    "湖南": "hn", "福建": "fj", "辽宁": "ln", "重庆": "cq", "陕西": "sx2", "黑龙江": "hlj"
}
data_set["province"] = data_set["province"].map(province_map)  # 将汉字映射为缩写
province_data = pd.get_dummies(data_set["province"], prefix="province")  # 省份独热编码
model_data = pd.get_dummies(data_set["model"], prefix="model")           # 车型独热编码
bodyType_data = pd.get_dummies(data_set["bodyType"], prefix="bodyType")  # 车身独热编码
data_set = data_set[["regYear", "regMonth", "salesVolume", "date", "popularity", "newsReplyVolume", "carCommentVolume"]]
data_set = data_set.join(province_data)    # 合并省份的独热编码
data_set = data_set.join(model_data)       # 合并车型的独热编码
data_set = data_set.join(bodyType_data)    # 合并车身的独热编码

data_set.to_csv("../data/train_data.csv", index=False)  # 将处理后数据写入文件