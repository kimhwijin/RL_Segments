import sys
sys.path.append("/home/hjkim/RL_TimeSegment")

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import TimeXer, TimesNet, layers, TCN, LSTM

def get_default_backbone(d_in, d_model, seq_len, backbone):
    if backbone.lower() == 'timesnet':
        return TimesNet.Model(
            enc_in=d_in,
            seq_len=seq_len,
            pred_len=0,
            top_k=3,
            d_model=d_model,
            d_ff=32,
            e_layers=3
        )
    elif backbone.lower() == 'timexer':
        return TimeXer.Model(
            enc_in=d_in,
            seq_len=seq_len,
            d_model=d_model,
            d_ff=32,
            e_layers=3,
        )
    elif backbone.lower() == 'tcn':
        return TCN.Model(
            enc_in = d_in,
            seq_len=seq_len,
            num_channels = [d_model]*3,
        )
    elif 'rnn' in backbone.lower():
        return LSTM.Model(
            enc_in = d_in,
            d_model = d_model,
            e_layers = 2,
            pooling = backbone.lower().split("_")[-1]
        )
    elif backbone.lower() == '':
        return nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        

