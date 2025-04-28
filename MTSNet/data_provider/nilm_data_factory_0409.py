from data_provider.nilm_data_loader_0409 import Dataset_Custom, Dataset_Pred,Dataset_NILM_minute
from torch.utils.data import DataLoader
from scipy.fft import fft
import numpy as np
import pandas as pd
import os

data_dict = {
    # 'custom': Dataset_Custom,
    'fridge_house2_shorten50_scale': Dataset_NILM_minute,  # 将NILM数据集迁移过来，把任务改为 单变量预测单变量，即 总功率分解单个电器功率
    'DW_h2_shorten50_scale': Dataset_NILM_minute, # 将NILM数据集迁移过来，把任务改为 单变量预测单变量，即 总功率分解单个电器功率
    'microwave_house2_shorten50_scale': Dataset_NILM_minute,  # 将NILM数据集迁移过来，把任务改为 单变量预测单变量，即 总功率分解单个电器功率
    'kettle_house2_shorten30_scale': Dataset_NILM_minute,  # 将NILM数据集迁移过来，把任务改为 单变量预测单变量，即 总功率分解单个电器功率
    'WM_h2_shorten50_scale': Dataset_NILM_minute,
    'DW_scale': Dataset_NILM_minute,
    'FRG_scale': Dataset_NILM_minute
}

def data_provider(args, flag):
    Data = data_dict[args.data]  # args.data = fridge_house2_all
    timeenc = 0 if args.embed != 'timeF' else 1    # args.embed默认为'timeF'，则timeenc为1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(  # return seq_x, seq_y, seq_x_mark, seq_y_mark
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        dataset=args.dataset,  # redd
        redd_train_data_path=args.redd_train_data_path,  # redd
        redd_val_data_path=args.redd_val_data_path,  # redd
        redd_test_data_path=args.redd_test_data_path  # redd
    )
    # data_set = Data(  # return seq_x, seq_y, seq_x_mark, seq_y_mark
    #     root_path=args.root_path,
    #     data_path=args.data_path,
    #     flag=flag,
    #     size=[args.seq_len, args.label_len, args.pred_len],
    #     features=args.features,
    #     target=args.target,
    #     timeenc=timeenc,
    #     freq=freq,
    # )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,  # return seq_x, seq_y, seq_x_mark, seq_y_mark
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True)
    return data_set, data_loader

