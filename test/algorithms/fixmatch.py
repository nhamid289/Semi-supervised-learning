from semilearn.algorithms.fixmatch import SSLFixMatch
from semilearn.algorithms import SSLAlgorithm
from semilearn.datasets.cv_datasets import Cifar10
from semilearn.utils.data import SSLDataLoader
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from semilearn import get_net_builder, get_config

from train import ssl_train

config = {
    'algorithm': 'fixmatch',
    'net': 'vit_tiny_patch2_32',
    'use_pretrain': True,
    'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',

    # optimization configs
    'epoch': 1,
    'num_train_iter': 4096,
    'num_eval_iter': 16,
    'num_log_iter': 64,
    'optim': 'AdamW',
    'lr': 5e-4,
    'layer_decay': 0.5,
    'batch_size': 16,
    'eval_batch_size': 16,


    # dataset configs
    'dataset': 'cifar10',
    'num_labels': 40,
    'num_lbl': 4,
    'num_ulbl': None,
    'num_classes': 10,

    'img_size': 32,
    'crop_size':32, # same as img size

    'crop_ratio': 0.875,
    'data_dir': './data',
    'ulb_samples_per_class': None,

    # algorithm specific configs
    'use_hard_label': True,
    'uratio': 2,
    'ulb_loss_ratio': 1.0,
    'lambda_u': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
    "num_workers": 2,
}
config = get_config(config)



algorithm = SSLFixMatch(lambda_u=1.0)

Net = get_net_builder(config.net, from_name=False)
model = Net(pretrained=config.use_pretrain, pretrained_path=config.pretrain_path, num_classes=10)

lr = 0.0005
mom = 0.9
nesterov=True
weight_decay=0.0005

optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=mom, nesterov=nesterov, weight_decay=weight_decay)

labels_per_class = 8
data = Cifar10(num_lbl=labels_per_class)

lbl_batch_size = 16
uratio = 2

train_loader = SSLDataLoader(data.get_lbl_dataset(), data.get_ulbl_dataset(), lbl_batch_size=lbl_batch_size, ulbl_batch_size=lbl_batch_size*uratio)

nepochs=128
ssl_train(model, algorithm, optimizer, train_loader, nepochs=nepochs, device="cuda")

now = datetime.now()

time = now.strftime("%H:%M")

torch.save(model.state_dict(),
           f"saved_models/fixmatch/cifar10-{now.date()},{time}-nlbpc:{labels_per_class}"+
           f"lbs:{lbl_batch_size}-ur:{uratio}-ne:{nepochs}-lr:{lr}-"+
           f"mom:{mom}-nest:{nesterov}-wd:{weight_decay}.pth")

