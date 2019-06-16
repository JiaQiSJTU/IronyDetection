
import os, time
import numpy as np

import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data import Data, IronyDataset, IronyDatasetTest
from model import *
from config import MODELCONFIG, TRAINCONFIG
from metric import *

import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

no_cuda = False

device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
print("Device:{}".format(device))

# train set
if TRAINCONFIG.retrain:
    if os.path.exists(TRAINCONFIG.MODELPATH_A):
        os.remove(TRAINCONFIG.MODELPATH_A)
    if os.path.exists(TRAINCONFIG.MODELPATH_B):
        os.remove(TRAINCONFIG.MODELPATH_B)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load and pading sequence batches
# dynamic padding: seqeuences are padded to the maximum length of mini-batch sequences
def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
    # print("batch length:{}".format(lengths))
    features = torch.FloatTensor((list(map(lambda x: x[2], sorted_batch))))
    return sequences_padded, labels, lengths, features

# def train(model, dim_model, train_loader, loss_func, optimizer):
#     model.train()
#     for batch in train_loader:
#         x, y, lengths = batch
#         # print(lengths)
#         optimizer.zero_grad()
#         # !!TOCHECK add positional encoding here or other place??
#         x = data.positional_encoding(x, dim_model)
#         out = model(x)
#         loss = loss_func(out, y)
#         loss.backward()
#         optimizer.step()
#     return loss

data = Data()
trainA, validA, trainB, validB, test = data.load_train_valid_test()

# LOAD DATA
train_feature, test_feature = data.load_seq_feats()

train_A_data = IronyDataset(trainA, train_feature, data.seq_to_tensor)
train_B_data = IronyDataset(trainB, train_feature, data.seq_to_tensor)

valid_A_data = IronyDataset(validA, train_feature, data.seq_to_tensor)
valid_B_data = IronyDataset(validB, train_feature, data.seq_to_tensor)

test_data = IronyDatasetTest(test, test_feature, data.seq_to_tensor)

print("feature:", train_feature.shape, test_feature.shape)

# padding_tensor = data.load_padding_tensor()


# parameters
# batch_size = 3450
batch_size = 128


# TASK A
# TRAIN LOAD
train_A_loader = DataLoader(train_A_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
valid_A_loader = DataLoader(valid_A_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

n_class_A = 2

# MODEL A
# bilstm = BLSTM(MODELCONFIG.dim_model, MODELCONFIG.n_hidden)
attention_layer = MultiHeadedAttention(MODELCONFIG.n_heads, MODELCONFIG.dim_model, MODELCONFIG.dropout)
ff_layer_A = PositionwiseFeedForward(MODELCONFIG.dim_model, n_class_A)
model_A = SelfAttenClassifier(bilstm=BLSTM(MODELCONFIG.dim_model, MODELCONFIG.n_hidden),
                              encoder=Encoder(MODELCONFIG.dim_model, EncoderLayer(MODELCONFIG.dim_model, attention_layer, ff_layer_A),
                                              MODELCONFIG.n_encoder_layer),
                              classifier=SoftMax(MODELCONFIG.d_ff, n_class_A))

if torch.cuda.is_available():
    model_A = model_A.to(device)

# LOSS S
# optimizer = torch.optim.Adam(model_A.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model_A.parameters(), lr=0.01, momentum=0.9)

import time
# loss_function_A = F.nll_loss
if torch.cuda.is_available():
    # loss_function_A = nn.CrossEntropyLoss().to(device)
    loss_function_A = nn.NLLLoss().to(device)
else:
    # loss_function_A = nn.CrossEntropyLoss()
    loss_function_A = nn.NLLLoss()

time_p, tr_acc_array, va_acc_array, ts_acc, loss_p = [], [], [], [], []

best_valid_acc_A = 0
best_state_A = model_A.state_dict()
print("\nTask A")
for epoch in range(MODELCONFIG.epoch_A):
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()


    t_start = time.perf_counter()

    train_loss = train(model_A, MODELCONFIG.dim_model, train_A_loader, loss_function_A, optimizer, device)
    train_acc = binary_acc(model_A, MODELCONFIG.dim_model, train_A_loader, device)
    valid_acc = binary_acc(model_A, MODELCONFIG.dim_model, valid_A_loader, device)

    if valid_acc >= best_valid_acc_A:
        best_state_A = model_A.state_dict()
        best_valid_acc_A = valid_acc

    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()

    t_end = time.perf_counter()
    time_p.append(t_end)
    loss_p.append(train_loss)
    tr_acc_array.append(train_acc)
    va_acc_array.append(valid_acc)

    print('Epoch: {:03d}, Acc: {:.8f}, Valid acc:{:.8f}, Duration: {:.2f}, Train loss:{}'.
          format(epoch, train_acc, valid_acc, t_end - t_start, train_loss))

# SAVE MODEL
if not os.path.exists("model/"):
    os.makedirs("model/")

torch.save(model_A.state_dict(), TRAINCONFIG.MODELPATH_A)

# PREDICT A
DESPATH = "res/"
if not os.path.exists(DESPATH):
    os.makedirs(DESPATH)
des_path = os.path.join(DESPATH, "pred_A_1.txt")
predict(model_A, MODELCONFIG.dim_model, TRAINCONFIG.MODELPATH_A, test_loader, des_path, device)


# TASK Bß
#TRAIN LOAD
train_B_loader = DataLoader(train_B_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
valid_B_loader = DataLoader(valid_B_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

n_class_B = 4

# MODEL B
ff_layer_B = PositionwiseFeedForward(MODELCONFIG.dim_model, n_class_B)
# model_B = SelfAttenClassifier(encoder=Encoder(MODELCONFIG.dim_model, EncoderLayer(MODELCONFIG.dim_model, attention_layer, ff_layer_B),
#                                               MODELCONFIG.n_encoder_layer),
#                               classifier=SoftMax(MODELCONFIG.d_ff, n_class_B))

model_B = SelfAttenClassifier(bilstm=BLSTM(MODELCONFIG.dim_model, MODELCONFIG.n_hidden),
                              encoder=Encoder(MODELCONFIG.dim_model, EncoderLayer(MODELCONFIG.dim_model, attention_layer, ff_layer_B),
                                              MODELCONFIG.n_encoder_layer),
                              classifier=SoftMax(MODELCONFIG.d_ff, n_class_B))

if torch.cuda.is_available():
    model_B = model_B.to(device)
else:
    model_B = model_B.to(device)

# loss_function_B = nn.NLLLoss(MODELCONFIG.label_weights)
if torch.cuda.is_available():
    # loss_function_B = nn.CrossEntropyLoss(MODELCONFIG.label_weights).to(device)
    loss_function_B = nn.NLLLoss().to(device)
else:
    # loss_function_B = nn.CrossEntropyLoss(MODELCONFIG.label_weights)
    loss_function_B = nn.NLLLoss(MODELCONFIG.label_weights)

time_p, tr_acc_array, va_acc_array, ts_acc, loss_p = [], [], [], [], []

# running epoches
# optimizer = torch.optim.Adam(model_B.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model_B.parameters(), lr=0.01, momentum=0.9)
best_valid_acc_B = 0
best_state_B = model_B.state_dict()
print("\nTask B")
for epoch in range(MODELCONFIG.epoch_B):
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()

        train_loss = train(model_B, MODELCONFIG.dim_model, train_B_loader, loss_function_B, optimizer, device)
        train_acc = binary_acc(model_B, MODELCONFIG.dim_model, train_B_loader, device)
        valid_acc = binary_acc(model_B, MODELCONFIG.dim_model, valid_B_loader, device)

        if valid_acc >= best_valid_acc_B:
            best_state_B = model_B.state_dict()
            best_valid_acc_B = valid_acc

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()
        time_p.append(t_end)
        loss_p.append(train_loss)
        tr_acc_array.append(train_acc)
        va_acc_array.append(valid_acc)

        print('Epoch: {:03d}, Acc: {:.8f}, Valid acc:{:.8f}, Duration: {:.2f}, Train loss:{}'.
              format(epoch, train_acc, valid_acc, t_end - t_start, train_loss))

# save model params
torch.save(model_B.state_dict(), TRAINCONFIG.MODELPATH_B)

# PREDICT B
des_path = os.path.join(DESPATH, "pred_B_1.txt")
predict(model_B, MODELCONFIG.dim_model, TRAINCONFIG.MODELPATH_B, test_loader, des_path, device)

