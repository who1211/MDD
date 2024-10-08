import numpy as np
import pandas as pd
import os.path as osp

import torch
import torch.nn.functional as func
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

from Model import GCN
from Dataset import ConnectivityData

#kfold

def GCN_train(loader):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def GCN_test(loader):
    model.eval()

    pred = []
    label = []
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = func.cross_entropy(output, data.y)
        loss_all += data.num_graphs * loss.item()
        pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    epoch_sen = tp / (tp + fn + 1e-7)
    epoch_spe = tn / (tn + fp + 1e-7)
    epoch_acc = (tn + tp) / (tn + tp + fn + fp)



    return epoch_sen, epoch_spe, epoch_acc, loss_all / len(val_dataset)


dataset = ConnectivityData('./data')
#

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = np.genfromtxt(osp.join(dataset.raw_dir, 'SLabels.txt'), encoding='utf-8', delimiter=',', skip_header=1)
eval_metrics = np.zeros((skf.n_splits, 3))

for n_fold, (train_val, test) in enumerate(skf.split(labels, labels)):

    model = GCN(dataset.num_features, dataset.num_classes, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_val_dataset, test_dataset = dataset[train_val.tolist()], dataset[test.tolist()]
    train_val_labels = labels[train_val]
    train_val_index = np.arange(len(train_val_dataset))

    train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True, stratify=train_val_labels)
    train_dataset, val_dataset = train_val_dataset[train.tolist()], train_val_dataset[val.tolist()]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    min_v_loss = np.inf
    for epoch in range(50):
        t_loss = GCN_train(train_loader)
        val_sen, val_spe, val_acc, v_loss = GCN_test(val_loader)
        test_sen, test_spe, test_acc, _ = GCN_test(test_loader)

        if min_v_loss > v_loss:
            min_v_loss = v_loss
            best_val_acc = val_acc
            best_test_sen, best_test_spe, best_test_acc = test_sen, test_spe, test_acc
            torch.save(model.state_dict(), 'best_model_%02i.pth' % (n_fold + 1))
            print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, '
                  'TEST SPE: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_acc, best_test_acc,
                                            best_test_sen, best_test_spe))

    eval_metrics[n_fold, 0] = best_test_sen
    eval_metrics[n_fold, 1] = best_test_spe
    eval_metrics[n_fold, 2] = best_test_acc

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'ACC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))





