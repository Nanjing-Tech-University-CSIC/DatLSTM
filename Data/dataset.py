
import numpy as np
import torch
from torch.utils.data import Dataset


bohai_datas_path = r'../Train_Datas/BohaiSea_samples.npy'
bohai_labels_path = r'../Train_Datas/BohaiSea_labels.npy'



########################
class BHDataset(Dataset):

    def __init__(self, args, split):

        super(BHDataset, self).__init__()

        # 加载数据
        self.datas = np.load(bohai_datas_path)   # (14591, 10, 1, 64, 64)
        # 加载标签
        self.targets = np.load(bohai_labels_path)
        # train : valid : test = 7:1:2
        if split == 'train':
            # 取训练数据
            self.datas = self.datas[:10213]
            self.targets = self.targets[:10213]
        elif split == 'valid':
            # 取验证集数据
            self.datas = self.datas[10213:11672]
            self.targets = self.targets[10213:11672]
        else:
            # 取测试集数据
            self.datas = self.datas[11672:]
            self.targets = self.targets[11672:]
        # 初始化
        # self.input_size = args.input_size  # default(64, 64)
        #
        # self.num_frames_input = args.num_frames_input  # default 10
        # self.num_frames_output = args.num_frames_output  # default 10
        # self.num_frames_total = args.num_frames_input + args.num_frames_output  # 总共是20帧
        print('Loaded X: {}'.format(len(self.datas)))
        print('Loaded Y: {}'.format(len(self.targets)))
        # 打印输出数据点的数目
        print('Loaded {} {} samples'.format(self.__len__(), split))

    def __getitem__(self, index):

        inputs = torch.from_numpy(self.datas[index])     # [10,1,64,64]
        targets = torch.from_numpy(self.targets[index])  # [10,1,64,64]
        return inputs,targets

    def __len__(self):
        # 返回数据集的长度
        return self.datas.shape[0]

