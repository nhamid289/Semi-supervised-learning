from semilearn.algorithms.fixmatch import SSLFixMatch
from semilearn.datasets.cv_datasets import Cifar10

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score


from semilearn import get_dataset, get_data_loader, get_net_builder, get_config, get_optimizer

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


algorithm = SSLFixMatch(config)

Net = get_net_builder(config.net, from_name=False)
model = Net()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


data = Cifar10(config)

# dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir, include_lb_to_ulb=config.include_lb_to_ulb)
# train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
# train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
# eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size)


def ssl_train(model, algorithm, optimizer, config, train_lb_loader, train_ulb_loader):


    model.cuda(config.gpu)
    model.train()

    for epoch in range(config.epoch):
        epoch_bar = tqdm(zip(train_lb_loader, train_ulb_loader),
                         desc=f"Epoch {epoch+1}/{config.epoch}")


        for i, (data_lb, data_ulb) in enumerate(epoch_bar, start=0):

            if i >= config.num_train_iter:
                break

            torch.cuda.synchronize()
            optimizer.zero_grad()


            loss = algorithm.train_step(model, **algorithm.process_batch(**data_lb, **data_ulb))

            loss.backward()

            if (i % (config.num_train_iter // config.num_log_iter) == 0):
                epoch_bar.set_postfix(loss=loss.item(), iteration=i)

    torch.cuda.synchronize()

ssl_train(model, algorithm, optimizer, config, data.get_lbl_loader(), data.get_ulbl_loader())




def ssl_eval(model, config, eval_loader, return_logits=False):
    model.eval()

    total_loss = 0.0
    total_num = 0.0
    y_true = []
    y_pred = []
    y_probs = []
    y_logits = []
    with torch.no_grad():
        for data in eval_loader:
            x = data['x_lb']
            y = data['y_lb']

            if isinstance(x, dict):
                x = {k: v.cuda(config.gpu) for k, v in x.items()}
            else:
                x = x.cuda(config.gpu)
            y = y.cuda(config.gpu)

            num_batch = y.shape[0]
            total_num += num_batch

            logits = model(x)['logits']

            loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.append(logits.cpu().numpy())
            y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.item() * num_batch

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        # top5 = top_k_accuracy_score(y_true, y_pred, k=5)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        print("accuracy: ", top1)
        # print("accuracy top 5: ", top5)
        print("balanced-accuracy: ", balanced_top1)
        print("recall: ", recall)
        print("f1: ", F1)

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        print('confusion matrix:\n' + np.array_str(cf_mat))

        model.train()

        # eval_dict = {eval_dest+'/loss': total_loss / total_num, eval_dest+'/top-1-acc': top1, eval_dest+'/top-5-acc': top5,
        #              eval_dest+'/balanced_acc': balanced_top1, eval_dest+'/precision': precision, eval_dest+'/recall': recall, eval_dest+'/F1': F1}
        # if return_logits:
        #     eval_dict[eval_dest+'/logits'] = y_logits
        # return eval_dict

# ssl_eval(model, config, eval_loader)