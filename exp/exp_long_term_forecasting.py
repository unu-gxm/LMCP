from distutils.command.config import config
from sklearn.decomposition import PCA
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from thop import profile
from thop import clever_format

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # args.patch_len
        self.patch_size = [args.patch_len]
        print(args.patch_len)
        self.masks = []
        for patch_len in self.patch_size:
            self.masks.append(self._get_mask(patch_len))
        # self.patch_size = [48,8]
        # self.masks = []
        # for patch_len in self.patch_size:
        #     self.masks.append(self._get_mask(patch_len))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

 ######  计算效率分析
        # #summary(model, input_size=(self.args.batch_size, 336, self.args.enc_in), device="cuda", verbose=2, depth=2)
        inputs= torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).cuda()
        batch_key_frequency = torch.randn(self.args.batch_size,self.args.seq_len,4 ).cuda()
        flops, macs = profile(model.cuda(), inputs=(inputs,batch_key_frequency, batch_key_frequency, batch_key_frequency), verbose=False)
        flops, macs = clever_format([flops, macs], "%.3f")
        print(f"FLOPS: {flops}, MACs: {macs}")
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.3fM" % (total / 1e6))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_mask(self,patch_len):
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // patch_len
        N = self.args.seq_len // patch_len
        print(N,self.args.seq_len,patch_len)
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks
    
    def _get_mask_2(self):
        dtype = torch.float32
        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len

        mask_base = torch.eye(L, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        mask0 = torch.eye(L, device=self.device, dtype=dtype)
        mask0.view(self.args.c_out, N, self.args.c_out, N).diagonal(dim1=0, dim2=2).fill_(1)
        mask0 = mask0.unsqueeze(0).unsqueeze(0) - mask_base
        mask1 = torch.kron(torch.ones(self.args.c_out, self.args.c_out, device=self.device, dtype=dtype), 
                            torch.eye(N, device=self.device, dtype=dtype))
        mask1 = mask1.unsqueeze(0).unsqueeze(0) - mask_base
        mask2 = torch.ones((1, 1, L, L), device=self.device, dtype=dtype) - mask1 - mask0 - mask_base
        masks = torch.cat([mask0, mask1, mask2], dim=0)  # [3, 1, L, L]
        return masks

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                outputs, _ ,_,_,  relship= self.model(batch_x, self.masks, is_training=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        contrastive_loss = nn.MSELoss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 在 train() 的数据加载后添加检查：
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 检查输入数据
                assert not torch.isnan(batch_x).any(), "Input batch_x contains NaN!"
                assert not torch.isinf(batch_x).any(), "Input batch_x contains inf!"
                assert not torch.isnan(batch_y).any(), "Target batch_y contains NaN!"
                # encoder - decoder
                outputs,dec_out_time,dec_out_vari, moe_loss ,  relship= self.model(batch_x, self.masks, is_training=True)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                alpha = 0.05
                lossmae = torch.nn.functional.l1_loss(outputs, batch_y)
                loss = criterion(outputs, batch_y) + alpha * moe_loss+lossmae+contrastive_loss(dec_out_time, dec_out_vari)


                # print("loss----------------------------------",loss)
                train_loss.append(loss.item())
                # ----------------------------------------------------------------------------------------------------------修改
                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            ######  显存占用统计
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为MB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # 转换为MB
            print(f" Allocated {allocated:.2f} MB, Max Allocated {max_allocated:.2f} MB, Reserved {reserved:.2f} MB")

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            relships=[]
            mean_corrs=[]
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                outputs, _ ,_,_,  relship= self.model(batch_x, self.masks, is_training=False)


                # batch_xrle=batch_x.transpose(2,1)
                # x = batch_xrle - batch_xrle.mean(dim=2, keepdim=True)
                # x = x / (batch_xrle.std(dim=2, keepdim=True) + 1e-8)
                # # Step 2: 计算通道间相关性矩阵 [batch, C, C]
                # corr_matrix = torch.matmul(x, x.transpose(1, 2)) / x.shape[2]
                # # Step 3: 去掉负相关（只保留正相关）
                # corr_matrix = torch.relu(corr_matrix)  # 负相关置 0
                # mean_corrs.append(corr_matrix)
                relships.append(outputs)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                input_ = batch_x
                pred = outputs
                true = batch_y

                inputs.append(input_)
                preds.append(pred)
                trues.append(true)

            # mean_corrs = torch.cat(mean_corrs, dim=0)
            # relships = torch.cat(relships, dim=0)
            #
            # mean_corr=mean_corrs.mean(dim=(0))
            # mean_corr = mean_corr.cpu().numpy()  # 先移到CPU再转成numpy
            # df = pd.DataFrame(mean_corr)
            # name = self.args.data
            # df.to_csv(f"mean_corr{name}mean_corr2.csv", index=False)
            #
            # relship = relships.transpose(2, 1)
            # x_np = relship.reshape(-1, 96).detach().cpu().numpy()
            #
            # pca = PCA(n_components=2)
            # x_pca = pca.fit_transform(x_np)
            # relship = torch.tensor(x_pca).reshape(-1, 7, 2)  # 恢复形状
            #
            # relship=relship.mean(dim=(0))
            # # relship = relship.cpu().numpy()  # 先移到CPU再转成numpy
            # df = pd.DataFrame(relship)
            # name = self.args.data
            # df.to_csv(f"PCA_{name}_matrix.csv", index=False)
            # print('PCAsuccess')
            # print(mean_corr.shape, relship.shape)
        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        # result save
        folder_path = './results/' + setting + '/'
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

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'input.npy', inputs)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
