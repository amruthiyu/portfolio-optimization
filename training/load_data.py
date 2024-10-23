import pandas as pd
import csv
import numpy as np
import os
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

code_num = 928
trade_day = 400
fts = 4
def load_EOD_data():
    f = open(r'C:\Users\amrut\Desktop\stock-portfolio-main-copy\data\result(928)_label.csv')
    df = pd.read_csv(f)
    data = df.iloc[:, 4:8].values
    eod_data = data.reshape(code_num, trade_day, fts)
    data_label = df.iloc[:, 6] # 将取label的列转为取收盘价！第4列
    # 数据集为：开盘价，最高，最低，收盘价，成交量，成交额；还有5日，10日，20日，30日均线
    data_label = round(data_label / data_label.shift(1), 4).values  # 获取价格相对向量yt
    # print('data_label_2', )  # data_label
    # print(eod_data[0])
    ground_truth = data_label.reshape(code_num, trade_day)

    return eod_data, ground_truth

def get_batch(eod_data, gt_data, offset, seq_len):

    return eod_data[:, offset:offset + seq_len, :], \
           gt_data[:, offset + seq_len]


def get_industry_adj():
    adj = np.mat(pd.read_csv(open(r'C:\Users\amrut\Desktop\stock-portfolio-main-copy\data\industry_adj(928).csv'), header=None))
    return adj
