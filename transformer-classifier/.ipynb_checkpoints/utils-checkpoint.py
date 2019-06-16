# +
import os
import json
from json import JSONDecoder
import numpy as np
import copy
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# -

def draw_curves(arr, y_name="loss"):
    color = cm.viridis(0.7)
    f, ax = plt.subplots(1,1)
    
    epoches = [i for i in range(len(arr))]
    ax.plot(epoches, arr, color=color)

    ax.set_xlabel('epoches')
    ax.set_ylabel('loss')

    plt.show()


# +
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        dataList=data['data']
    return dataList

def load_seq_feats(filename):
    with open(filename,'r')as f:
        test_feature=f.read()
    test_feature=JSONDecoder().decode(test_feature)
    return test_feature


# +
def binary_acc(model,loader, device=torch.device('cpu')):
    model.eval()
    num_corrects = 0
    for data in loader:
        x, y, lengths, addition_feats = data
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        addition_feats = addition_feats.to(device)
        
#         x = positional_enc(x, dim_model)
        maxlen = x.size(1)
        mask = torch.arange(maxlen)[None, :].to(device) < lengths[:, None].to(device)
        
        with torch.no_grad():
            pred,seq_feats = model(x, mask, addition_feats)
#         print(torch.max(pred, 1)[1].view(y.size()).data)
#         print(y.data)
        num_corrects += (torch.max(pred, 1)[1].view(y.size()).data == y.data).sum()
    return num_corrects.item() / len(loader.dataset)

def train(model, train_loader, loss_func, optimizer, device=torch.device('cpu')):
    model.train()
    for batch in train_loader:
        
        x, y, lengths, addition_feats = batch
        
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.to(device)
        addition_feats = addition_feats.to(device)
        
#         x = positional_enc(x, dim_model)
        
        optimizer.zero_grad()
        
        maxlen = x.size(1)
        mask = torch.arange(maxlen)[None, :].to(device) < lengths[:, None].to(device)

        out, seq_feats = model(x, mask, addition_feats)
        
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
    return loss
