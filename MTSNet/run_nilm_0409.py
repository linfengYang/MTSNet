import argparse
import torch
from experiments.exp_nilm_forecasting_0409 import Exp_NILM_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='DW_scale',
                        help='dataset type')  
    parser.add_argument('--root_path', type=str, default='./data/NILM/',
                        help='root path of the data file') 
    parser.add_argument('--data_path', type=str, default='DW_h2_all.csv',
                        help='data csv file') 

    # 添加的语句变量
    parser.add_argument('--dataset', type=str, default='UKDALE', help='UKDALE or REDD')
    parser.add_argument('--redd_train_data_path', type=str, default='REDD_0626/DW/DW_train.csv',
                        help='根据电器目录选择训练集文件') 
    parser.add_argument('--redd_val_data_path', type=str, default='REDD_0626/DW/DW_val.csv',
                        help='根据电器目录选择验证集文件') 
    parser.add_argument('--redd_test_data_path', type=str, default='REDD_0626/DW/DW_test.csv',
                        help='根据电器目录选择测试集文件')  

    parser.add_argument('--features', type=str, default='MS',  
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')  # OT
    parser.add_argument('--freq', type=str, default='s', 
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # TSLANet ----- 0701
    parser.add_argument('--ASB', default=True)  # type=str2bool,
    parser.add_argument('--emb_dim', type=int, default=64, help='dimension of model')  # ---##64
    parser.add_argument('--adaptive_filter', default=True)  # type=str2bool,
    parser.add_argument('--depth', type=int, default=3, help='num of layers')
    parser.add_argument('--patch_size', type=int, default=64, help='size of patches')  # ---64
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout value')

    # PatchTST
    parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9], help='kernel size list')

    parser.add_argument('--Fd_model', type=int, default=64, help='dimension of model')  # 64

    parser.add_argument('--patch_len', type=int, nargs='+', default=[16, 8],
                        help='patch high')  # [8,4]5 [16, 8] [10, 5]

    parser.add_argument('--period', type=int, nargs='+', default=[24, 12],
                        help='period list')  # [12,8] 10 [24, 12] [16, 8]

    parser.add_argument('--Fdropout', type=float, default=0.3, help='dropout value')

    parser.add_argument('--stride', type=int, nargs='+', default=None, help='stride')

    parser.add_argument('--Fd_ff', type=int, default=256, help='dimension of fcn')  # 256

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=500,
                        help='input sequence length')  
    parser.add_argument('--label_len', type=int, default=0,
                        help='start token length') 
    parser.add_argument('--pred_len', type=int, default=500,
                        help='prediction sequence length') 

    # model define
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1,
                        help='output size')  
    parser.add_argument('--d_model', type=int, default=64,
                        help='dimension of model') 
    parser.add_argument('--n_heads', type=int, default=12, help='num of heads') 
    parser.add_argument('--e_layers', type=int, default=3,
                        help='num of encoder layers') 
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1521, help='dimension of fcn') 
    parser.add_argument('--factor', type=int, default=5,
                        help='attn factor')  
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--Dropout', type=float, default=0.01, help='dropout')  
    parser.add_argument('--embed', type=str, default='timeF',  
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation') 
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', default=False,
                        help='whether to predict unseen future data')  # ---

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')  
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=80, help='train epochs') 

    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  # 8 ----
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')  
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')  # ---
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')  
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')  # ---

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',  # -------
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False,
                        help='whether to use channel_independence mechanism')
    parser.add_argument('--channel_independences', type=int, default=1,
                        help='whether to use channel_independence mechanism')
    parser.add_argument('--class_strategy', type=str, default='projection',
                        help='projection/average/cls_token')  # ------

    args = parser.parse_args()  #
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train':  # See Figure 8 of our paper, for the detail
        # Exp = Exp_NILM_Forecast  # Exp_Long_Term_Forecast_Partial
        print('------')
    else:  # MTSF: multivariate time series forecasting
        Exp = Exp_NILM_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model,
                args.data,
                args.features,
                args.dataset,
                args.emb_dim,
                args.depth,
                args.patch_size,
                args.adaptive_filter,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_ff,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)  # --

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.dataset,
            args.emb_dim,
            args.depth,
            args.patch_size,
            args.adaptive_filter,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
