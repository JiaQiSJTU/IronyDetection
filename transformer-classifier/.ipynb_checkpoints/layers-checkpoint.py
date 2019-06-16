import os
import json
from json import JSONDecoder
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# return x+pe
def positional_enc(x, dim_model, max_len=5000):
    base = 10000
    sentence_len = x.size(1)
    pe_vec = torch.zeros(max_len, dim_model)
    p = torch.arange(0., max_len).unsqueeze(1)
    frac = torch.exp(torch.arange(0., dim_model, 2) * -(math.log(10000.0) / dim_model)) 
    pe_vec[:,0::2] = torch.sin(p)
    pe_vec[:,1::2] = torch.cos(p)
    pe_vec = pe_vec.unsqueeze(0)
    return x.float() + pe_vec[:,:sentence_len]

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    dim_key = query.size(-1)
    attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_key)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(attn, dim = -1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value), attn_weights

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # make sure input word embedding dimension divides by the number of desired heads
        assert dim_model % num_heads == 0
        # assume dim of key,query,values are equal
        self.dim_qkv = dim_model // num_heads
        
        self.dim_model = dim_model
        self.num_h = num_heads
        self.w_q = nn.Linear(dim_model, dim_model) # self.w_qs = nn.Linear(d_model, n_head * d_k) 
        self.w_k = nn.Linear(dim_model, dim_model) 
        self.w_v = nn.Linear(dim_model, dim_model)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, query, key, value, mask=None):
        n_batch = query.size(0)
        
        if mask is not None:
#             mask = mask.unsqueeze(1)
              mask = mask.view(n_batch,mask.size(1),1,1).expand(n_batch,mask.size(1),self.num_h,self.num_h)
        
        # linear projections: dim_model => num_h x dim_k 
        query = self.w_q(query).view(n_batch, -1, self.num_h, self.dim_qkv)
        key = self.w_k(key).view(n_batch, -1, self.num_h, self.dim_qkv)
        value = self.w_v(value).view(n_batch, -1, self.num_h, self.dim_qkv)
        
        # Apply attention on all the projected vectors in batch 
        x, self.attn = scaled_dot_product_attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # Concat(head1, ..., headh) 
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.num_h * self.dim_qkv)
        
        x = nn.Linear(dim_model, dim_model, bias=False)(x)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu # bert uses gelu instead

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class AddNorm(nn.Module):
    def __init__(self, size, dropout, eps=1e-6):
        super(AddNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x + self.dropout(sublayer(norm))

class EncoderLayer(nn.Module):
    def __init__(self, size, attention, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.self_atten = attention
        self.add_norm_1 = AddNorm(size, dropout)
        self.add_norm_2 = AddNorm(size, dropout)
        self.size = size

    def forward(self, x, mask=None):
        output = self.add_norm_1(x, lambda x: self.self_atten(x, x, x, mask))
        output = self.add_norm_2(output, self.feed_forward)
        return output
