import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_NILM_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # 对数据进行统一的标准化
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))  # 读取csv数据
        ''' csv数据格式如下
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # 下面两条语句表示从总数据长度20*30*24中选择60%(12*30*24)用于训练，其余各20%用于验证及测试  其实有点问题这里，写法不标准
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]   # train: border1s[0]=0  val: border1s[1]=12 * 30 * 24 - self.seq_len
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 编号从0开始，.columns[1:]表示从第二列读到最后一列
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        '''
        根据self.features的值选择特征。
            如果是'M'或'MS'，选择除日期外的所有特征； [多变量预测多变量,多变量预测单变量]
            如果是'S'，只选择目标特征。 [S：单变量预测单变量]
        '''
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]  # 当前某个模式(train/val/test)下的第一列时间数据
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:  # 如果 args.embed != 'timeF'，就会把时间编码为 month，day，weekday，hour 四个数
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq) # 根据传入的 freq 对时间戳进行解析
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# class Dataset_NILM_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='*****.csv',
#                  target='OT', scale=False, timeenc=0, freq='t'):  # 原先scale=True
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:  # size=[args.seq_len, args.label_len, args.pred_len]
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()  # 标准化------
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         # border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         # border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         num_train = int(len(df_raw) * 0.6)  # 取60%做训练集
#         num_test = int(len(df_raw) * 0.2)  # 取20%做验证集
#         num_vali = len(df_raw) - num_train - num_test  # 取剩下的20%做测试集
#         # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]  # 这里的seq_len怎么理解？
#         # border2s = [num_train, num_train + num_vali, len(df_raw)]
#
#         # ---
#         border1s = [0, num_train, num_train + num_vali]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         # ---
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]  # 取第二列至最后一列的数据，即抛弃第一列date数据
#             # ---
#             # cols_data = df_raw.columns[1:2]  # 取第二列至最后一列的数据
#             # ---
#             df_data = df_raw[cols_data]  # (1723874,2)
#         elif self.features == 'S':  # 如果为S，表示单变量预测单变量，表示预测一条序列未来的数据，只用OT这一列
#             df_data = df_raw[[self.target]]
#
#         if self.scale:  # 是否标准化---
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#         # print('----:',self.scale)
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:  #  走这条分支---
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)  # （1034324,6）
#
#         # self.data_x = data[border1:border2]  # (344872,2)
#         # self.data_y = data[border1:border2]  # (344872,2)
#         # temp_x =  self.data_x
#
#         # ---
#         self.data_x = data[border1:border2,0:1]  # (344872,1)
#         self.data_y = data[border1:border2,1:2]
#         # ---
#         self.data_stamp = data_stamp  # (344872,6)
#
#     def __getitem__(self, index):  #
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         # r_begin = s_end - self.label_len
#         # r_end = r_begin + self.label_len + self.pred_len  # = s_begin+seq_len+pred_len
#
#         # seq_x = self.data_x[s_begin:s_end]  # seq_len
#         # seq_y = self.data_y[r_begin:r_end]  # pred_len + label_len
#         # seq_x_mark = self.data_stamp[s_begin:s_end]
#         # seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         # ---
#         seq_x = self.data_x[s_begin:s_end]  # seq_len
#         seq_y = self.data_y[s_begin:s_end]  # pred_len
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[s_begin:s_end]
#         # ---
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
class Dataset_NILM_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='*****.csv',
                 target='OT', scale=False, timeenc=0, freq='t',
                 dataset='',redd_train_data_path='', redd_val_data_path='', redd_test_data_path=''):  # 原先scale=True
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:  # size=[args.seq_len, args.label_len, args.pred_len]
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.dataset = dataset  # ----
        self.redd_train_data_path = redd_train_data_path  # redd训练集----
        self.redd_val_data_path = redd_val_data_path  # redd验证集----
        self.redd_test_data_path = redd_test_data_path  # redd测试集----

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # 标准化------

        # 使用UK_DALE数据集（同一house的数据划分训练集、验证集及测试集）------------------
        if self.dataset =='UKDALE':
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                              self.data_path))
            num_train = int(len(df_raw) * 0.6)  # 取60%做训练集
            num_test = int(len(df_raw) * 0.2)  # 取20%做验证集
            num_vali = len(df_raw) - num_train - num_test  # 取剩下的20%做测试集

            # ---
            border1s = [0, num_train, num_train + num_vali]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            # ---
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]  # 取第二列至最后一列的数据，即抛弃第一列date数据
                df_data = df_raw[cols_data]  # (1723874,2)
            elif self.features == 'S':  # 如果为S，表示单变量预测单变量，表示预测一条序列未来的数据，只用OT这一列
                df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
            # print('----:',self.scale)
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:   # 走这条分支---
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)  # （1034324,6）

            # ---
            self.data_x = data[border1:border2,0:1]  # (344872,1)
            self.data_y = data[border1:border2,1:2]
            # ---
            self.data_stamp = data_stamp  # (344872,6)

        # 使用REDD数据集（使用house[2, 3]的数据作为训练集、验证集，使用house1作为测试集） ------------------
        elif self.dataset =='REDD':
            if self.set_type == 0:  # 训练集
                df_raw = pd.read_csv(os.path.join(self.root_path,self.redd_train_data_path))
                cols_data = df_raw.columns[1:]  # 取第二列至最后一列的数据，即抛弃第一列date数据
                df_train_data = df_raw[cols_data]  # (1723874,2)
                data = df_train_data.values
            elif self.set_type == 1:  # 验证集
                df_raw = pd.read_csv(os.path.join(self.root_path,self.redd_val_data_path))
                cols_data = df_raw.columns[1:]  # 取第二列至最后一列的数据，即抛弃第一列date数据
                df_val_data = df_raw[cols_data]  # (1723874,2)
                data = df_val_data.values
            elif self.set_type == 2:  # 测试集
                df_raw = pd.read_csv(os.path.join(self.root_path,self.redd_test_data_path))
                cols_data = df_raw.columns[1:]  # 取第二列至最后一列的数据，即抛弃第一列date数据
                df_test_data = df_raw[cols_data]  # (1723874,2)
                data = df_test_data.values

            df_stamp = df_raw[['date']][:]  # 由于df_raw就是对应训练或测试的数据集，因此取全部行
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:  # 走这条分支---
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)  # （1034324,6）
            # ---
            self.data_x = data[:, 0:1]  # 由于data就是对应训练或测试的数据集，因此取全部行
            self.data_y = data[:, 1:2]
            # ---
            self.data_stamp = data_stamp  # (344872,6)

    def __getitem__(self, index):  #
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len  # = s_begin+seq_len+pred_len

        # seq_x = self.data_x[s_begin:s_end]  # seq_len
        # seq_y = self.data_y[r_begin:r_end]  # pred_len + label_len
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        # ---
        seq_x = self.data_x[s_begin:s_end]  # seq_len
        seq_y = self.data_y[s_begin:s_end]  # pred_len
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[s_begin:s_end]
        # ---
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='***.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='***.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 从第一列到最后一列？？？？有个问题：为什么输入要包含OT这一列
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
