# DatLSTM: A new model for spatio-temporal prediction

This repository contains the official PyTorch implementation of the following paper:

Sea Surface Temperature Prediction Using ConvLSTM-based Model with Deformable Attention

# Introduction
  ![Architecture](/Architecture.png)


This model skillfully combines Convolutional Neural Networks (CNNs) with Recurrent Neural Networks (RNNs), enabling it to simultaneously capture spatiotemporal dependencies within a single computational framework. 
To overcome the limitation that CNNs primarily capture local spatial information, in this paper, we propose a novel model named DatLSTM, which integrates a deformable attention transformer (DAT) module into the ConvLSTM framework, enhancing its ability to process more complex spatial relationships effectively. Specifically, the DAT module adaptively focuses on salient features in space, while the ConvLSTM further captures the temporal dependencies of spatial correlations in the SST data. In doing so, DatLSTM can adaptively capture complex spatiotemporal dependencies between preceding and current states within ConvLSTM. To evaluate the performance of the DatLSTM model, we conduct short-term SST forecasts with a forecast lead time ranging from 1 to 10 days in the Bohai Sea region and compare its efficacy against several benchmark models, including ConvLSTM, PredRNN, TCTN, and SwinLSTM. The experimental results show that our proposed model outperforms these models in terms of multiple evaluation metrics for short-term SST prediction. 
The proposed model offers a new predictive learning method for improving the accuracy of spatiotemporal predictions in various domains such as meteorology, oceanography, and climate science.


## Overview
- `DatLSTM.py` contains the model with a multiple DatLSTM cell.
- `dataset.py` contains training and validation dataloaders.
- `functions.py` contains train and test functions.
- `utils.py` contains functions for model training and evaluation
- `configs.py` contains train and test the model parameters
- `trainDatLSTM.py` is the core file for training pipeline.
- `test.py` is a file for a quick test.


## Requirements
- python >= 3.8
- torch == 1.11.0
- torchvision == 0.12.0
- numpy
- matplotlib
- skimage == 0.19.2
- timm == 0.4.12
- einops == 0.4.1

## Acknowledgment
These codes are based on [SwinLSTM] (https://github.com/SongTang-x/SwinLSTM/).
We extend our sincere appreciation for their valuable contributions.
