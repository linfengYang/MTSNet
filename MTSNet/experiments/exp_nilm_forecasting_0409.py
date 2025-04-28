from data_provider.nilm_data_factory_0409 import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch_dct as dct
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Exp_NILM_Forecast(Exp_Basic):  # 传入args
    def __init__(self, args):
        super(Exp_NILM_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def count_parameters(self,only_trainable=True):
        if only_trainable:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        #model_optim = AdaBelief(self.model.parameters(), lr=self.args.learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,rectify=True)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()  # 能否把损失函数改掉------
        return criterion

    def compute_dtw_loss(self,pred, target):
        # pred and target are both of size [32, 500, 1]
        B, T, D = pred.shape
        dtw_loss = 0.0

        for b in range(B):
            D_matrix = torch.cdist(pred[b], target[b], p=2)  # Compute pairwise distance matrix
            cost_matrix = torch.zeros_like(D_matrix)
            cost_matrix[0, 0] = D_matrix[0, 0]

            # Dynamic programming to compute the DTW path
            for i in range(1, T):
                cost_matrix[i, 0] = cost_matrix[i - 1, 0] + D_matrix[i, 0]
            for j in range(1, T):
                cost_matrix[0, j] = cost_matrix[0, j - 1] + D_matrix[0, j]
            for i in range(1, T):
                for j in range(1, T):
                    cost_matrix[i, j] = D_matrix[i, j] + torch.min(torch.min(cost_matrix[i - 1, j], cost_matrix[i, j - 1]), cost_matrix[i - 1, j - 1])


            dtw_loss += cost_matrix[-1, -1] / T  # Normalize by sequence length
        return dtw_loss / B  # Average over batch

    def compute_temporal_loss(self,pred, target):
        # pred and target are both of size [32, 500, 1]
        temporal_loss = F.mse_loss(pred[:, 1:, :] - pred[:, :-1, :], target[:, 1:, :] - target[:, :-1, :])
        return temporal_loss

    def dilate_loss(self,pred, target, alpha=0.5, gamma=0.001):
        """
        pred: [32, 500, 1]  (model prediction)
        target: [32, 500, 1]  (ground truth)
        alpha: weighting between DTW loss and temporal loss
        gamma: scaling factor for temporal loss
        """
        dtw_loss = self.compute_dtw_loss(pred, target)
        temporal_loss = self.compute_temporal_loss(pred, target)

        total_loss = alpha * dtw_loss + (1 - alpha) * gamma * temporal_loss
        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        huber_loss = nn.HuberLoss(delta=0.6, reduction="mean")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # ---每次取完数据后，令输入x为总功率这一列，输出y为OT这一列
                # batch_x = batch_x[:, :, -2:-1]
                # batch_y = batch_y[:, :, -1:]
                # ---
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # else:
                # if self.args.output_attention:
                #     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                # else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #
                # # ---
                # loss = torch.mean(torch.mean(torch.pow(outputs - batch_y, 2), dim=1))
                # ---
                pred = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)  # (32,96,1)取最后一列，即数据集中OT这一列
                true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # (32,96,1)
                # loss = criterion(outputs, batch_y)

                # 损失函数
                loss = torch.mean(torch.mean(torch.pow(pred - true, 2), dim=1)).to(self.device)  # 损失函数也可以进行优化
                #
                # pre_error = torch.abs(pred - true).to(self.device)
                # correct_mask = (pred < true)
                # amplify_loss = torch.mean((correct_mask * pre_error) ** 2).to(self.device)

                # Correct_mask = (pred > true)
                # Smooth_loss = torch.mean((Correct_mask * pred) ** 2).to(self.device)

                # loss = base_loss + 0.2*amplify_loss + 0.1 * Smooth_loss
                # loss = base_loss + 0.03*amplify_loss
                # loss = huber_loss(pred, true)
                # loss = self.dilate_loss(pred,true)

                total_loss.append(loss)
        total_loss = torch.tensor(total_loss).cpu()  # ----新加
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        sum = self.count_parameters(self.model)
        print(sum)
        # 数据是否需要标准化？-----在nilm_data_loader使用了StandardScaler()进行标准化
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        if self.args.dataset == 'UKDALE':
            path = os.path.join(self.args.checkpoints + 'UKDALE/', setting)
        else:
            path = os.path.join(self.args.checkpoints + 'REDD/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # fre_loss = nn.L1Loss()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()  # 优化器可进行调整为AdaBelief
        criterion = self._select_criterion()
        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()  # batch_x_mark, batch_y_mark 位置编码？
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # (32,600,1)  (32,96,2)  为什么输入batch_x要把OT这一列包含进去
                batch_y = batch_y.float().to(self.device)  # (32,600,1)  (32,144,2) 为什么输出batch_y要把总功率这一列包含进去
                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                batch_x_mark = batch_x_mark.float().to(self.device)  # (32,600,6) (32,96,6)
                batch_y_mark = batch_y_mark.float().to(self.device)  # (32,600,6) (32,144,6)

                # decoder input  0408-------
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # (32,600,1)  (32,96,2)
                # print(dec_inp.shape)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(
                    self.device)  # (32,648,1)  (32,144,2)
                # print(dec_inp.shape)
                # encoder - decoder
                # if self.args.use_amp:  # use_amp   default=False
                #     with torch.cuda.amp.autocast():
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #
                #         f_dim = -1 if self.args.features == 'MS' else 0
                #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #         loss = criterion(outputs, batch_y)
                #         train_loss.append(loss.item())
                # else:
                #     if self.args.output_attention:  # False
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #     else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                     batch_y_mark)  # (32,96,2)  forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

                f_dim = -1 if self.args.features == 'MS' else 0  # 根据任务MS(多变量预测单变量)来设置输出维度-----
                pred = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)  # (32,96,1)取最后一列，即数据集中OT这一列
                true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # (32,96,1)
                # loss = criterion(outputs, batch_y)

                # 损失函数
                loss = torch.mean(torch.mean(torch.pow(pred - true, 2), dim=1)).to(self.device)  # 损失函数也可以进行优化

                # f_o = dct.dct(outputs.permute(0, 2, 1)).permute(0, 2, 1)
                # f_y = dct.dct(batch_y.permute(0, 2, 1)).permute(0, 2, 1)
                train_loss.append(loss.item())
                # loss += torch.mean(torch.mean(torch.abs(f_o - f_y), dim=1)).to(self.device)
                #
                # pre_error = torch.abs(pred - true).to(self.device)
                # correct_mask = (pred < true)
                # amplify_loss = torch.mean((correct_mask * pre_error) ** 2).to(self.device)


                # Correct_mask = (pred > true)
                # Smooth_loss = torch.mean((Correct_mask * pred) ** 2).to(self.device)

                # loss = base_loss + 0.2*amplify_loss + 0.1 * Smooth_loss

                # loss = base_loss + 0.03*amplify_loss  # 0.05
                # loss = huber_loss(pred,true)

                # loss = self.dilate_loss(pred, true)




                # ---
                # train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # if self.args.use_amp:
                #     scaler.scale(loss).backward()
                #     scaler.step(model_optim)
                #     scaler.update()
                # else:
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)  # 本质上还是验证，但这么做的目的是判断当前模型对测试集的有效性
            #
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)  # --
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)  # ----是否真需要调整学习率？？？

            torch.cuda.empty_cache()  # 添加的加快训练的代码

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)
            best_model_path = path + '/' + 'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path))  # 由于CUDA设备序列化不对应添加了,map_location=self.device

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if self.args.dataset == 'UKDALE':
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/UKDALE/' + setting, 'checkpoint.pth')))
            else:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/REDD/' + setting, 'checkpoint.pth'),map_location='cuda:0'))
                # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/REDD/redd_ds', 'checkpoint.pth')))


        preds = []
        trues = []
        mains = []  # 总功率-------
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # print(len(test_loader))

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # --- 每次取完数据后，令输入x为总功率这一列，输出y为OT这一列
                # batch_x = batch_x[:, :, -2:-1]
                # batch_y = batch_y[:, :, -1:]
                # ---
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # else:
                #     if self.args.output_attention:
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #
                #     else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                main_batch_x = batch_x[:, -self.args.pred_len:, f_dim:].to(self.device)  # 总功率-------
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                main_batch_x = main_batch_x.detach().cpu().numpy()
                # if test_data.scale and self.args.inverse:
                #     shape = outputs.shape
                #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y


                preds.append(pred)
                trues.append(true)
                mains.append(main_batch_x)  # 总功率-------
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        mains = np.array(mains) # 总功率-------
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mains = mains.reshape(-1, mains.shape[-2], mains.shape[-1]) # 总功率-------
        print('test shape:', preds.shape, trues.shape)

        # result save
        if self.args.dataset=='UKDALE':
            folder_path = './results/' + 'UKDALE/'+setting + '/'
        else:
            folder_path = './results/' + 'REDD/'+setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + '0615_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + '0615_pred.npy', preds)
        np.save(folder_path + '0615_true.npy', trues)
        np.save(folder_path + '0615_main.npy', mains)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:  # 输入已经经过标准化，需要对输出反标准化
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return