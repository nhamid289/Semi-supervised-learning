
config = {

    'save_dir': './saved_models/usb_cv/script/',

    # algorithm specific configs
    'algorithm': 'fixmatch',
    'hard_label': True,
    'uratio': 2,
    'ulb_loss_ratio': 1.0,
    'include_lb_to_ulb': False,

    # optimization configs
    'epoch': 1,
    'num_train_iter': 4096,
    'num_eval_iter': 128,
    'num_log_iter': 64,
    'optim': 'AdamW',
    'lr': 5e-4,
    'layer_decay': 0.5,
    'batch_size': 16,
    'eval_batch_size': 16,


    # dataset configs
    'data_dir': './data',
    'dataset': 'cifar10',
    'num_labels': 40,
    'num_classes': 10,
    'img_size': 32,
    'crop_ratio': 0.875,
    'ulb_samples_per_class': None,

    'ulb_num_labels': 49600,
    'lb_imb_ratio': 1,
    'lb_imb_ratio': 1,



    'net': 'vit_tiny_patch2_32',
    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
    "num_workers": 2,

    'amp': False,
}

