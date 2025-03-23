import semilearn
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from config import config

eval_config = {
    'use_pretrain': True,
    'pretrain_path': 'saved_models/usb_cv/fixmatch_cifar10_40_0/latest_model.pth',
}

config = {**config, **eval_config}

config = get_config(config)

algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir, include_lb_to_ulb=config.include_lb_to_ulb)

eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size, drop_last=False, shuffle=False)

trainer = Trainer(config, algorithm)

trainer.evaluate(eval_loader)

y_pred, y_logits = trainer.predict(eval_loader)