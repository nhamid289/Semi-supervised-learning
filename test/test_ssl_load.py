from semilearn.algorithms.fixmatch import SSLFixMatch
from semilearn.datasets.cv_datasets import Cifar10
from semilearn.utils.data import SSLDataLoader

from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score

from semilearn import get_net_builder, get_config

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



Net = get_net_builder(config.net, from_name=False)
model = Net(num_classes=10)
# model = Net(pretrained=config.use_pretrain, pretrained_path=config.pretrain_path, num_classes=10)

checkpoint = torch.load("saved_models/fixmatch/cifar10-2025-04-23,17:06-nlbpc:8lbs:16-ur:2-ne:2048-lr:0.0005-mom:0.9-nest:True-wd:0.0005.pth")
model.load_state_dict(checkpoint)

cifar10 = Cifar10()

eval_loader = DataLoader(cifar10.get_eval_dataset(), batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"

def ssl_eval(model, eval_loader, device="cpu"):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_num = 0.0
    y_true = []
    y_pred = []
    y_probs = []
    y_logits = []

    with torch.no_grad():
        for X, _, _, _, y in eval_loader:
            X = X.to(device)
            y = y.to(device)


            num_batch = y.shape[0]
            total_num += num_batch

            logits = model(X)['logits']

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

ssl_eval(model, eval_loader=eval_loader, device=device)