from typing import Literal
from torch import nn
from .encoder import LogEncoder, MetricEncoder, TraceEncoder
import torch


class GatedFusion(nn.Module):
    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return output

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100, num_class=5):
        super(ConcatFusion, self).__init__()
        self.linear_x_out = nn.Linear(output_dim, output_dim)
        self.linear_y_out = nn.Linear(output_dim, output_dim)
        self.linear_z_out = nn.Linear(output_dim, output_dim)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.squeeze = nn.AdaptiveAvgPool2d((1, output_dim))
        self.clf = nn.Linear(output_dim, num_class)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        fc_out = self.fc_out(output)
        x_out = self.linear_x_out(x)
        y_out = self.linear_y_out(y)
        z_out = self.linear_z_out(z)
        output = torch.stack(
            (
                x,
                y,
                z,
                self.sigmoid(x_out),
                self.sigmoid(y_out),
                self.sigmoid(z_out),
                self.sigmoid(fc_out),
            ),
            dim=1,
        )
        output = self.squeeze(output)
        output = output.squeeze(dim=1)
        output = self.clf(output)
        return x_out, y_out, z_out, output


class AdaFusion(nn.Module):
    def __init__(
        self,
        kpi_num: int,
        invoke_num: int,
        instance_num: int,
        max_len: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        layer_num: int,
        dropout: float,
        num_class: int,
        device: str,
    ) -> None:
        super().__init__()
        self.log_encoder = LogEncoder(
            max_len, d_model, nhead, d_ff, layer_num, dropout, device
        )
        self.metric_encoder = MetricEncoder(
            kpi_num,
            instance_num,
            max_len,
            d_model,
            nhead,
            d_ff,
            layer_num,
            dropout,
            device,
        )
        self.trace_encoder = TraceEncoder(invoke_num, d_model, nhead, d_ff, dropout)

        self.clf = nn.Sequential(
            nn.Linear(d_model, num_class),
        )
        self.clf_cat = nn.Sequential(
            nn.Linear(d_model * 3, num_class),
        )

        self.concat_fusion = ConcatFusion(d_model * 3, d_model, num_class)

    def forward(self, x_list):
        x_log = self.log_encoder(x_list[0])
        x_metric = self.metric_encoder(x_list[1])
        x_trace = self.trace_encoder(x_list[2])
        return self.concat_fusion(x_metric, x_log, x_trace)


class ExperimentModel(nn.Module):
    def __init__(
        self,
        kpi_num: int,
        invoke_num: int,
        instance_num: int,
        max_len: int,
        d_model: int,
        nhead: int,
        d_ff: int,
        layer_num: int,
        dropout: float,
        num_class: int,
        device: str,
    ) -> None:
        super().__init__()
        self.log_encoder = LogEncoder(
            max_len, d_model, nhead, d_ff, layer_num, dropout, device
        )
        self.metric_encoder = MetricEncoder(
            kpi_num,
            instance_num,
            max_len,
            d_model,
            nhead,
            d_ff,
            layer_num,
            dropout,
            device,
        )
        self.trace_encoder = TraceEncoder(invoke_num, d_model, nhead, d_ff, dropout)

        self.clf = nn.Sequential(
            nn.Linear(d_model, num_class),
        )

        self.clf_cat = nn.Sequential(
            nn.Linear(d_model * 3, num_class),
        )

        self.use_modal = "all"

    def set_use_modal(self, modal: Literal["log", "metric", "trace", "all"] = "all"):
        self.use_modal = modal

    def forward(self, x_list):
        if self.use_modal == "all":
            return self.__fusion_forward__(x_list)
        else:
            return self.__single_forward__(x_list)

    def __fusion_forward__(self, x_list):
        x_log = self.log_encoder(x_list[0])
        x_metric = self.metric_encoder(x_list[1])
        x_trace = self.trace_encoder(x_list[2])
        return self.clf_cat(torch.cat([x_log, x_metric, x_trace], dim=-1))

    def __single_forward__(self, x_list):
        modal = self.use_modal
        hiddens = None
        if modal == "log":
            hiddens = self.log_encoder(x_list[0])
        elif modal == "metric":
            hiddens = self.metric_encoder(x_list[1])
        elif modal == "trace":
            hiddens = self.trace_encoder(x_list[2])
        else:
            raise Exception("unknown modal")
        return self.clf(hiddens)
