

import sys
import os

# Add the parent directory to the sys.path
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import semilearn
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer

config = {
    'algorithm': 'fixmatch',
    'net': 'vit_tiny_patch2_32',
    'use_pretrain': True,
    'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',
    'save_dir': '../saved_models/usb_cv/',

    'resume': False,
    'load_path': './saved_models/usb_cv//fixmatch_cifar10_40_0/latest_model.pth',

    # optimization configs
    'epoch': 1,
    'num_train_iter': 5000,
    'num_eval_iter': 500,
    'num_log_iter': 50,
    'optim': 'AdamW',
    'lr': 5e-4,
    'layer_decay': 0.5,
    'batch_size': 16,
    'eval_batch_size': 16,


    # dataset configs
    'dataset': 'cifar10',
    'num_labels': 40,
    'num_classes': 10,
    'img_size': 32,
    'crop_ratio': 0.875,
    'data_dir': '../data',
    'ulb_samples_per_class': None,

    # algorithm specific configs
    'hard_label': True,
    'uratio': 2,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
    "num_workers": 2,

    'amp': False,
}

config = get_config(config)


algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)


dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir, include_lb_to_ulb=config.include_lb_to_ulb)
train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size)

trainer = Trainer(config, algorithm)
trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)

trainer.evaluate(eval_loader)

y_pred, y_logits = trainer.predict(eval_loader)

