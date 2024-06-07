from torch import nn
import torch
import numpy as np
from torch.nn import functional as F


class LogEncoder(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        layer_num: int,
        dropout: float,
        device,
    ) -> None:
        super().__init__()
        self.position_tensor = self.__create_position_tensor__(max_len, d_model).to(
            device
        )
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layer_num)

        self.globalAvgPooling1D = nn.AdaptiveAvgPool1d(1)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def __create_position_tensor__(self, max_len, d_model):
        angle_rads = np.arange(max_len).reshape(-1, 1) / np.power(
            10000, (2 * (np.arange(d_model).reshape(1, -1) // 2)) / np.float32(d_model)
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return torch.tensor(angle_rads, dtype=torch.float32).reshape(
            1, max_len, d_model
        )

    def __position_embedding__(self, seq_len):
        return self.position_tensor[:, :seq_len, :]

    def forward(self, inputs):
        r"""
        return (batch_size, d_model)
        """
        seq_len = inputs.size(dim=1)
        inputs += self.__position_embedding__(seq_len)
        inputs = self.encoder(inputs)
        hiddens = self.globalAvgPooling1D(inputs.transpose(1, 2)).squeeze(dim=2)
        hiddens = self.layernorm(hiddens)
        hiddens = self.dropout1(hiddens)
        return hiddens


class MetricEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        instance_num: int,
        max_len: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        layer_num: int,
        dropout: float,
        device,
    ) -> None:
        super().__init__()

        # instance -> microservice
        self.expand = nn.Sequential(
            nn.Linear(instance_num, d_model),
            nn.ReLU(),
            nn.Linear(d_model, instance_num),
        )
        self.layernorm1 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)

        # project feature
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # learn ts feature
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layer_num)

        # ts -> final repr
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # initialize
        self.proj.apply(
            lambda module: (
                nn.init.kaiming_normal_(module.weight)
                if isinstance(module, nn.Linear)
                else None
            )
        )

        # position embedding
        self.position_tensor = self.__create_position_tensor__(max_len, d_model).to(
            device
        )

    def __create_position_tensor__(self, max_len, d_model):
        angle_rads = np.arange(max_len).reshape(-1, 1) / np.power(
            10000, (2 * (np.arange(d_model).reshape(1, -1) // 2)) / np.float32(d_model)
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return torch.tensor(angle_rads, dtype=torch.float32).reshape(
            1, max_len, d_model
        )

    def __position_embedding__(self, seq_len):
        return self.position_tensor[:, :seq_len, :]

    def forward(self, inputs):
        r"""
        return (batch_size, d_model)
        """
        hiddens = self.agg_from_instance_to_microservice(inputs)
        hiddens = self.agg_from_ts(hiddens)
        return hiddens

    def agg_from_instance_to_microservice(self, inputs):
        r"""
        in  :   (batch_size, ins_num, ts_len, ?)
        out :   (batch_size, ts_len, ?)
        """
        inputs = inputs.transpose(1, 3)
        hiddens = inputs * F.sigmoid(
            self.expand(inputs.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True))
        )
        hiddens = hiddens.transpose(1, 3).sum(dim=1)
        hiddens = self.layernorm1(hiddens)
        hiddens = self.dropout1(hiddens)
        hiddens = self.proj(hiddens)
        return hiddens

    def agg_from_ts(self, hiddens):
        r"""
        in  :   (batch_size, ts_len, ?)
        out :   (batch_size, ?)
        """
        hiddens = hiddens + self.__position_embedding__(hiddens.size(dim=1))
        hiddens = self.encoder(hiddens)
        hiddens = F.adaptive_avg_pool1d(hiddens.transpose(1, 2), 1).squeeze(dim=2)
        hiddens = self.layernorm2(hiddens)
        hiddens = self.dropout2(hiddens)
        return hiddens


class TraceEncoder(nn.Module):
    def __init__(
        self, in_dim: int, d_model: int, nhead: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.project = nn.Linear(in_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.project(x)
        x = self.relu(x)
        x = self.layernorm(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
