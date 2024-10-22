import numpy as np
from sklearn.model_selection import train_test_split

from Data.dataset import BHDataset
from utils import *
import torch.nn as nn
from configs import get_args
from funAndUtils.functions import train, test
from torch.utils.data import DataLoader



class MSEL1Loss(nn.Module):
    def __init__(self, size_average=False, reduce=True, alpha=1.0):
        super(MSEL1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha

        self.mse_criterion = nn.MSELoss(size_average=self.size_average, reduce=self.reduce)
        self.l1_criterion = nn.L1Loss(size_average=self.size_average, reduce=self.reduce)

    def __call__(self, input, target):
        mse_loss = self.mse_criterion(input, target)
        l1_loss = self.l1_criterion(input, target)
        loss = mse_loss + self.alpha * l1_loss  # l1正则化mse损失
        return loss / 2.

def setup(args):
    global model

    if args.model == 'DatLSTM':
        from models.DatLSTM import DatLSTM
        model = DatLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                     in_chans=args.input_channels, embed_dim=args.embed_dim,
                     depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                     num_heads=args.num_heads, window_size=args.window_size,
                     n_head_channels= args.n_head_channels, n_groups=args.n_group,
                     proj_drop=args.proj_drop_rate, stride=2,
                     offset_range_factor=2, use_pe=True, dwc_pe=False,
                     no_off=False, fixed_pe=False, ksize=3, log_cpb=False).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 评价标准
    criterion = nn.MSELoss()

    # 加载数据集
    train_dataset = BHDataset(args, split='train')
    valid_dataset = BHDataset(args, split='valid')

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size,
                              num_workers=0, shuffle=True, pin_memory=True, drop_last=True)

    return model, criterion, optimizer, train_loader, valid_loader


def main():
    args = get_args()
    set_seed(args.seed)
    # 创建相应的目录
    cache_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    # 得到相关的训练参数
    model, criterion, optimizer, train_loader, valid_loader = setup(args)
    # 用于记录训练损失和验证集损失
    train_losses, valid_losses = [], []

    best_metric = (0, float('inf'), float('inf'))

    for epoch in range(args.epochs):

        start_time = time.time()
        train_loss = train(args, logger, epoch, model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        # 绘制损失曲线
        plot_loss(train_losses, 'train', epoch, args.res_dir, 1)

        # 每隔一定轮次进行一次验证
        if (epoch + 1) % args.epoch_valid == 0:

            # 记录验证损失，mse，ssim
            valid_loss, mse, ssim = test(args, logger, epoch, model, valid_loader, criterion, cache_dir,'valid')

            valid_losses.append(valid_loss)
            # 绘制验证集上的损失曲线
            plot_loss(valid_losses, 'valid', epoch, args.res_dir, args.epoch_valid)

            # 当前有最小的mse
            if mse < best_metric[1]:
                # 将当前模型做一个保存
                torch.save(model.state_dict(), f'{model_dir}/trained_model_state_dict')
                # 更新最佳的指标
                best_metric = (epoch, mse, ssim)
            # 日志进行输出
            logger.info(f'[Current Best] EP:{best_metric[0]:04d} MSE:{best_metric[1]:.4f} SSIM:{best_metric[2]:.4f}')

        # 答应当前epoch耗时
        print(f'Time usage per epoch: {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    main()
