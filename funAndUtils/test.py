import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.DatLSTM import DatLSTM
from configs import get_args
from Data.dataset import BHDataset
from funAndUtils.functions import test
from utils import set_seed, make_dir, init_logger

if __name__ == '__main__':
    # 获取参数配置对象
    args = get_args()
    set_seed(args.seed)
    # 创建相关结果目录
    cache_dir, model_dir, log_dir = make_dir(args)
    # 创建日志对象
    logger = init_logger(log_dir)

    # 初始化模型

    model = DatLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                    in_chans=args.input_channels, embed_dim=args.embed_dim,
                    depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                    num_heads=args.num_heads, window_size=args.window_size,
                    n_head_channels=args.n_head_channels, n_groups=args.n_group,
                    proj_drop=args.proj_drop_rate, stride=2,
                    offset_range_factor=2, use_pe=True, dwc_pe=False,
                    no_off=False, fixed_pe=False, ksize=3, log_cpb=False).to(args.device)
    criterion = nn.MSELoss()

    # 加载测试数据
    test_dataset = BHDataset(args,split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             num_workers=0, shuffle=False, pin_memory=True, drop_last=True)

    # 加载模型
    model.load_state_dict(torch.load('./results/model/trained_model_state_dict_bohai_DatLSTM'))
    # model.load_state_dict(torch.load('./results/model/trained_model_state_dict'))
    # 记录开始时间
    start_time = time.time()

    # 小样本进行测试实验，得到相关评价指标
    _, mse, ssim = test(args, logger, 0, model, test_loader, criterion, cache_dir,'test')
    # _, mae, rmse,r2 = test(args, logger, 0, model, test_loader, criterion, cache_dir, 'test')

    # 打印结果
    print(f'[Metrics]  MSE:{mse:.4f} SSIM:{ssim:.4f}')
    # print(f'[Metrics]  MAE:{mae:.4f} RMSE:{rmse:.4f} r2:{r2:.4f}')
    # 打印训练耗时

    print(f'Time usage per epoch: {time.time() - start_time:.0f}s')

