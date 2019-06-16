

import json
import numpy as np
from json import JSONDecoder
from config import TASK3
from math import log, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Data(object):
    def __init__(self):
        self.train_feature_path = TASK3.TRAIN_FEATURE
        self.test_feature_path = TASK3.TEST_FEATURE

        self.train_A_path = TASK3.TASK_A
        self.train_B_path = TASK3.TASK_B

        self.test_path = TASK3.TEST

        self.word_em_path = TASK3.WORDEMBEDDING
        self.POS_em_path = TASK3.POSEMBEDDING

        self.word_idx_path = TASK3.WORD2IDX
        self.pos_idx_path = TASK3.POS2IDX

    def load_data(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
            dataList = data['data']
        return dataList

    def load_dict(self, file):
        with open(file, 'r') as f:
            dictionary = json.load(f)
        return dictionary

    def load_seq_feats(self):
        with open(self.train_feature_path, 'r')as f:
            train_feature = f.read()
        train_feature = JSONDecoder().decode(train_feature)

        with open(self.test_feature_path, 'r')as f:
            test_feature = f.read()
        test_feature = JSONDecoder().decode(test_feature)
        return np.array(train_feature, dtype=np.float32), np.array(test_feature, dtype=np.float32)

    def load_train_valid_test(self):
        '''
        :return: trainA, validA, trainB, validB, test
        '''
        A = self.load_data(self.train_A_path)
        B = self.load_data(self.train_B_path)
        test = self.load_data(self.test_path)

        trainA, validA = A[:3450], A[3450:]
        trainB, validB = B[:3450], B[3450:]
        print("#train:{}\t#valid:{}\t#test:{}".format(len(trainA), len(validA), len(test)))
        return trainA, validA, trainB, validB, test

    def load_embedding(self):
        '''
        :return: word embedding [700], POS [28]
        '''
        return np.load(self.word_em_path), np.load(self.POS_em_path)

    def positional_encoding(self, x, dim_model, max_len=5000):
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
        return x.float() + pe_vec[:, :sentence_len]
        # return pe_vec[:, :sentence_len]

    def seq_to_tensor(self, raw_sample, dim_model=728):
        word_embedding, pos_embedding = self.load_embedding()
        word2idx, pos2idx = self.load_dict(self.word_idx_path), self.load_dict(self.pos_idx_path)
        seq_embed = torch.tensor([np.concatenate([word_embedding[word2idx[w]], pos_embedding[pos2idx[raw_sample["pos"][i]]]]) for i, w in enumerate(raw_sample["word"])])
        return seq_embed

    def load_padding_tensor(self):
        word_embedding, pos_embedding = self.load_embedding()
        word2idx, pos2idx = self.load_dict(self.word_idx_path), self.load_dict(self.pos_idx_path)
        # print(word_embedding[word2idx["PADDING"]])
        # print(pos_embedding[pos2idx["PADDING"]])
        # print(np.concatenate([word_embedding[word2idx["PADDING"]], pos_embedding[pos2idx["PADDING"]]]))
        # return torch.tensor(np.concatenate([word_embedding[word2idx["PADDING"]], pos_embedding[pos2idx["PADDING"]]]))
        return torch.tensor(np.array([0] * 728, dtype=np.float32))

    def feature_concatnate(self, x_batch, feature):
        x_batch = x_batch.numpy()
        new_x_batch = []
        for i in range(x_batch.shape[0]):
            new_x_batch_sent = []
            for j in range(x_batch.shape[1]):
                new_x_batch_sent.append(np.concatenate(x_batch[i][j], feature))
            new_x_batch.append(new_x_batch_sent)

        new_x_batch = np.array(new_x_batch)
        new_x_batch_tensor = torch.from_numpy(new_x_batch)
        return new_x_batch_tensor


class IronyDataset(Dataset):
    def __init__(self, raw_data, feature, transform=None):
        self.data = raw_data
        self.feature = feature
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        feature = self.feature[index]
        label = self.data[index]["label"]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, feature


class IronyDatasetTest(Dataset):
    def __init__(self, raw_data, feature, transform=None):
        self.data = raw_data
        self.feature = feature
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        feature = self.feature[index]
        # 随便设的
        label = 0

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, feature