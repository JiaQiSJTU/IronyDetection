

import torch
from data import Data, IronyDataset
import numpy as np
from math import log
from config import MODELCONFIG
from torch.autograd import Variable


data = Data()

padding_tensor = data.load_padding_tensor()


def positional_encoding(x, dim_model, device,  max_len=5000):
    # whether dropout?
    sentence_len = x.size(1)
    pe_vec = torch.zeros(max_len, dim_model)
    position = torch.arange(0., max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., dim_model, 2) * -(log(10000.0) / dim_model))
    pe_vec[:, 0::2] = torch.sin(position)
    pe_vec[:, 1::2] = torch.cos(position)
    # pe_vec[:, 0::2] = torch.sin(position * div_term)
    # pe_vec[:, 1::2] = torch.cos(position * div_term)
    pe_vec = pe_vec.unsqueeze(0)
    # print(x.shape, pe_vec.shape)
    return x.float() + pe_vec[:, :sentence_len].to(device)
    # return pe_vec[:, :sentence_len]

def get_mask_tensor(x_batch, padding_tensor):
    batch_padding = padding_tensor.repeat(x_batch.size(0), x_batch.size(1), 1).numpy()
    x_batch = x_batch.cpu().numpy()
    # print(batch_padding.shape)
    # print(x_batch.shape)
    mask_numpy=[]
    # print(x_batch[-1][-1])
    # print(x_batch[-1][-1].shape)
    # print(padding_tensor.numpy())
    for i in range(x_batch.shape[0]):
        mask_numpy_sen=[]
        for j in range(x_batch.shape[1]):
            # print(x_batch[i][j])
            # print(padding_tensor.numpy())
            if (x_batch[i][j] == padding_tensor.numpy()).all():
                # print("in")
                mask_numpy_sen.append(np.array([0] * 728, dtype=np.float32))
            else:
                mask_numpy_sen.append(np.array([1] * 728, dtype=np.float32))
        mask_numpy.append(mask_numpy_sen)
    mask_numpy = np.array(mask_numpy)
    # print(type(mask_numpy))
    mask_tensor = torch.from_numpy(mask_numpy)
    # print(mask_tensor.numpy())
    # print(mask_tensor.shape)
    return mask_tensor

    # print(torch.eq(x_batch, batch_padding).select(1, 2))
    # return torch.eq(x_batch, batch_padding)


def train(model, dim_model, train_loader, loss_func, optimizer, device):
    model.train()
    for batch in train_loader:
        if torch.cuda.is_available():
            batch = tuple(t.to(device) for t in batch)
        x, y, lengths, feature = batch
        # print("len:", lengths, x.shape, y.shape)
        mask_tensor = get_mask_tensor(x, padding_tensor)
        # print(lengths)
        optimizer.zero_grad()
        # x = positional_encoding(x, dim_model, device)
        if torch.cuda.is_available():
            out = model(x, feature, Variable(mask_tensor).to(device))
        else:
            out = model(x, feature, Variable(mask_tensor))
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
    return loss

def Acc(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

def binary_acc(model, dim_model, loader, device):
    model.eval()
    num_corrects = 0
    acc_list = []
    for data in loader:
        # x, y, lengths = batch
        if torch.cuda.is_available():
            data = tuple(t.to(device) for t in data)

        x, y, lengths, feature = data

        mask_tensor = get_mask_tensor(x, padding_tensor)
        # x = positional_encoding(x, dim_model, device)

        with torch.no_grad():
            if torch.cuda.is_available():
                pred = model(x, feature, Variable(mask_tensor).to(device))
            else:
                pred = model(x, feature, Variable(mask_tensor))

            acc_list.append(Acc(pred, y))
        #         print(torch.max(pred, 1)[1].view(y.size()).data)
        #         print(y.data)
        # num_corrects += (torch.max(pred, 1)[1].view(y.size()).data == y.data).sum()
    # return num_corrects.item() / len(loader.dataset)
    return float(sum(acc_list)) / len(acc_list)


def predict(model, dim_model, model_path, loader, des_path, device):
    model.eval()
    model.load_state_dict(torch.load(model_path))

    predict_labels = []

    for data in loader:
        if torch.cuda.is_available():
            data = tuple(t.to(device) for t in data)

        x, y, length, feature = data
        mask_tensor = get_mask_tensor(x, padding_tensor)
        # x = positional_encoding(x, dim_model, device)
        with torch.no_grad():
            pred = model(x, feature, Variable(mask_tensor).to(device))
            predict_labels.append(np.array(torch.argmax(pred).cpu()).tolist())

    with open(des_path, 'w') as f:
        for pre in predict_labels:
            f.write("{}\n".format(pre))
    # return predict_labels