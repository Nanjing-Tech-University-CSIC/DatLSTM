import os
from datetime import datetime

import numpy as np
import torch
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, visualize

scaler = amp.GradScaler()


def model_forward_single_layer(model, inputs, targets_len, num_layers):
    '''
    :param model: DatLSTM
    :param inputs: (batch_size,10,1,64,64)
    :param targets_len: 10
    :param num_layers: 网络层数
    :return: outputs
    '''
    outputs = []
    states = [None] * len(num_layers)

    inputs_len = inputs.shape[1]    # 10
    
    last_input = inputs[:, -1]

    # 对inputs序列进行循环
    for i in range(inputs_len - 1):
        output, states = model(inputs[:, i], states)
        outputs.append(output)

    #  对目标序列进行循环
    for i in range(targets_len):
        output, states = model(last_input, states)
        outputs.append(output)
        last_input = output

    return outputs


def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)

    for i in range(targets_len):
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs


def train(args, logger, epoch, model, train_loader, criterion, optimizer):
    model.train()
    # batch数量
    num_batches = len(train_loader)
    # 用于记录每个batch的损失
    losses = []

    # 对每一个batch训练
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # print("inputs_shape:", inputs.shape)
        # print("inputs_shape:", targets.shape)
        optimizer.zero_grad()
        # 转为float并用GPU训练 返回一个新的由转换后的inputs和targets组成的列表。
        # ([8, 10, 1, 64, 64])
        inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])

        # target_len 10
        targets_len = targets.shape[1]
        # print("targets_len:",targets_len)
        with autocast():
            if args.model == 'DatLSTM':
                outputs = model_forward_multi_layer(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            # inputs和warm-up的结果，predicts和targets计算损失
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)
            loss = criterion(outputs, targets_)


        # 将损失进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        # 每log_train打印一次损失
        if batch_idx and batch_idx % args.log_train == 0:
            logger.info(f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f}')

    return np.mean(losses)


def test(args, logger, epoch, model, test_loader, criterion, cache_dir,split):
    # 采用eval模式
    model.eval()
    # batch数目
    num_batches = len(test_loader)
    # 指标
    losses, mses, ssims = [], [], []
    # losses, maes, rmses, r2s= [], [], [],[]
    #  用于记录预测值
    total_pred = []

    total_true = []

    for batch_idx, (inputs, targets) in enumerate(test_loader):

        with torch.no_grad():
            inputs, targets = map(lambda x: x.float().to(args.device), [inputs, targets])

            targets_len = targets.shape[1]

            if args.model == 'DatLSTM':
                outputs = model_forward_multi_layer(model, inputs, targets_len, args.depths_down)

            outputs = torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()
            targets_ = torch.cat((inputs[:, 1:], targets), dim=1)

            losses.append(criterion(outputs, targets_).item())

            inputs_len = inputs.shape[1]

            outputs = outputs[:, inputs_len - 1:]   # (batch_size,targets_len,1,64,64)
            # 计算评价指标
            mse, ssim = compute_metrics(outputs, targets)
            # mae,rmse,r2 = compute_metrics(outputs, targets)
            # 记录指标
            mses.append(mse)
            ssims.append(ssim)
            # maes.append(mae)
            # rmses.append(rmse)
            # r2s.append(r2)
            if split == 'test':
                # 将每个批次的预测值和真实值添加到总列表中
                sample_true = (targets.cpu().numpy()).squeeze(axis=2)  # (8,10,64,64)
                sample_pred = (outputs.cpu().numpy()).squeeze(axis=2)  # (8,10,64,64)

                for b in range(args.test_batch_size):
                    total_true.append(sample_true[b])
                    total_pred.append(sample_pred[b])

            if batch_idx and batch_idx % args.log_valid == 0:
                logger.info(
                    f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MSE:{mse:.4f} SSIM:{ssim:.4f}')
                # logger.info(
                #     f'EP:{epoch:04d} BI:{batch_idx:03d}/{num_batches:03d} Loss:{np.mean(losses):.6f} MAE:{mae:.4f} RMSE:{rmse:.4f} R2{r2:.4f}')
                visualize(inputs, targets, outputs, epoch, batch_idx, cache_dir)

    if split == 'test':
        # 真实数据
        true_array = np.array(total_true)
        # 预测数据
        pred_array = np.array(total_pred)
        print('true_array.shape:', true_array.shape)
        print('pred_array.shape:', pred_array.shape)
        # 将数据进行保存
        test_data_save_dir = args.test_data_save_dir
        # 获取当前日期
        current_date = datetime.now().strftime("%Y%m%d")
        # 创建保存路径
        save_path = test_data_save_dir + '{}/'.format(current_date)
        # 确保路径存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存真实数据

        np.save(save_path + 'normalize_true_DatLSTM_' + args.test_year_time + '_' + args.data_name + '.npy',
                true_array)
        np.save(save_path + 'normalize_pred_DatLSTM_' + args.test_year_time + '_' + args.data_name + '.npy',
                pred_array)


    # 返回每一个指标的均值
    return np.mean(losses), np.mean(mses), np.mean(ssims)
    # return np.mean(losses), np.mean(maes), np.mean(rmses),np.mean(r2s)
