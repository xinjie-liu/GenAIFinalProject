import math
import pickle
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timeseries_synthesis.models.diffusion_models.imputers.S4Model import (
    S4Layer,
)

from timeseries_synthesis.utils.basic_utils import (
    get_autoencoder_config,
    get_denoiser_config,
    get_cltsp_config,
    get_dataset_config,
)

from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.utils import (
    MetaDataEncoder,
)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in, device):
    assert diffusion_step_embed_dim_in % 2 == 0
    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)
    return diffusion_step_embed.float()


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(
        self,
        res_channels,
        skip_channels,
        diffusion_step_embed_dim_out,
        in_channels,
        s4_lmax,
        s4_d_state,
        s4_dropout,
        s4_bidirectional,
        s4_layernorm,
        label_embed_dim=None,
    ):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        self.S41 = S4Layer(
            features=2 * self.res_channels,
            lmax=s4_lmax,
            N=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm,
        )

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(
            features=2 * self.res_channels,
            lmax=s4_lmax,
            N=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm,
        )

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        # the layer-specific fc for label embedding (conditional case)
        self.fc_label = nn.Linear(label_embed_dim, 2 * self.res_channels) if label_embed_dim is not None else None

    def forward(self, input_data):
        x, label_embed, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        h = self.conv_layer(h)

        h = self.S41(h.permute(2, 0, 1)).permute(1, 2, 0)

        # process label embedding
        if self.fc_label is not None:
            label_embed = self.fc_label(label_embed)
            label_embed = torch.einsum("blc->bcl", label_embed)  # (B,channels,L)
            h = h + label_embed

        h = self.S42(h.permute(2, 0, 1)).permute(1, 2, 0)

        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels :, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(
        self,
        res_channels,
        skip_channels,
        num_res_layers,
        diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out,
        in_channels,
        s4_lmax,
        s4_d_state,
        s4_dropout,
        s4_bidirectional,
        s4_layernorm,
        label_embed_dim=None,
        device=None,
    ):
        super(Residual_group, self).__init__()
        self.device = device
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                Residual_block(
                    res_channels,
                    skip_channels,
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                    in_channels=in_channels,
                    s4_lmax=s4_lmax,
                    s4_d_state=s4_d_state,
                    s4_dropout=s4_dropout,
                    s4_bidirectional=s4_bidirectional,
                    s4_layernorm=s4_layernorm,
                    label_embed_dim=label_embed_dim,
                )
            )

    def forward(self, input_data):
        noise, label_embed, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in, self.device
        )

        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, label_embed, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDenoiser_v1(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(SSSDenoiser_v1, self).__init__()
        self.config = config
        self.denoiser_config = get_denoiser_config(config=self.config)
        self.dataset_config = get_dataset_config(config=self.config)
        self.device = self.config.device

        self.in_channels = self.dataset_config.num_channels
        self.res_channels = self.denoiser_config.res_channels
        self.skip_channels = self.denoiser_config.skip_channels
        self.out_channels = self.dataset_config.num_channels
        self.num_res_layers = self.denoiser_config.num_res_layers
        self.diffusion_step_embed_dim_in = self.denoiser_config.diffusion_step_embed_dim_in
        self.diffusion_step_embed_dim_mid = self.denoiser_config.diffusion_step_embed_dim_mid
        self.diffusion_step_embed_dim_out = self.denoiser_config.diffusion_step_embed_dim_out
        self.s4_lmax = self.dataset_config.time_series_length
        self.s4_d_state = self.denoiser_config.s4_d_state
        self.s4_dropout = self.denoiser_config.s4_dropout
        self.s4_bidirectional = self.denoiser_config.s4_bidirectional
        self.s4_layernorm = self.denoiser_config.s4_layernorm

        self.label_embed_dim = self.res_channels
        print(OKBLUE + "Label embedding dimension = %d" % self.label_embed_dim + ENDC)

        self.init_conv = nn.Sequential(Conv(self.in_channels, self.res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(
            res_channels=self.res_channels,
            skip_channels=self.skip_channels,
            num_res_layers=self.num_res_layers,
            diffusion_step_embed_dim_in=self.diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=self.diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=self.diffusion_step_embed_dim_out,
            in_channels=self.in_channels,
            s4_lmax=self.s4_lmax,
            s4_d_state=self.s4_d_state,
            s4_dropout=self.s4_dropout,
            s4_bidirectional=self.s4_bidirectional,
            s4_layernorm=self.s4_layernorm,
            label_embed_dim=self.label_embed_dim,
            device=self.device,
        )

        self.final_conv = nn.Sequential(
            Conv(self.skip_channels, self.skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(self.skip_channels, self.out_channels),
        )

        # metadata encoder
        self.metadata_encoder = MetaDataEncoder(
            dataset_config=self.dataset_config,
            denoiser_config=self.denoiser_config,
            device=self.device,
        )

        T = 200
        beta_0 = 0.0001
        beta_T = 0.02

        self.diffusion_hyperparameters = self.calc_diffusion_hyperparams(
            T=T,
            beta_0=beta_0,
            beta_T=beta_T,
        )

        # print("diffusion_hyperparameters", self.diffusion_hyperparameters)

    def calc_diffusion_hyperparams(self, T, beta_0, beta_T):
        Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
        Alpha = 1 - Beta
        Alpha_bar = Alpha + 0
        Beta_tilde = Beta + 0
        for t in range(1, T):
            Alpha_bar[t] *= Alpha_bar[t - 1]
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
        Sigma = torch.sqrt(Beta_tilde)

        Beta = Beta.to(self.device)
        Alpha = Alpha.to(self.device)
        Alpha_bar = Alpha_bar.to(self.device)
        Sigma = Sigma.to(self.device)

        _dh = {}
        _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
            T,
            Beta,
            Alpha,
            Alpha_bar,
            Sigma,
        )
        diffusion_hyperparams = _dh
        return diffusion_hyperparams

    def prepare_training_input(self, train_batch):
        sample = train_batch["timeseries_full"].float().to(self.device)
        # print(sample.shape)
        assert sample.shape[1] == self.in_channels

        # discrete and continuous condition input
        discrete_label_embedding = train_batch["discrete_label_embedding"].float().to(self.device)
        if len(discrete_label_embedding.shape) == 2:
            discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
            discrete_label_embedding = discrete_label_embedding.repeat(1, sample.shape[2], 1)

        continuous_label_embedding = train_batch["continuous_label_embedding"].float().to(self.device)

        # side info
        L = sample.shape[2]
        batch_size = sample.shape[0]

        T, Alpha_bar = (
            self.diffusion_hyperparameters["T"],
            self.diffusion_hyperparameters["Alpha_bar"],
        )
        Alpha_bar = Alpha_bar.to(self.device)

        # diffusion step
        t = torch.randint(
            T,
            size=(batch_size, 1, 1),
        ).to(self.device)

        # noise and noisy data

        noise = torch.randn_like(sample).float().to(self.device)
        noisy_sample = torch.sqrt(Alpha_bar[t]) * sample + torch.sqrt(1 - Alpha_bar[t]) * noise

        denoiser_input = {
            "noisy_sample": noisy_sample,
            "noise": noise,
            "sample": sample,
            "discrete_cond_input": discrete_label_embedding,
            "continuous_cond_input": continuous_label_embedding,
            "diffusion_step": t,
        }

        return denoiser_input

    def forward(self, denoiser_input):
        noisy_sample = denoiser_input["noisy_sample"]
        B = noisy_sample.shape[0]
        K = noisy_sample.shape[1]  # K
        L = noisy_sample.shape[2]  # L

        cond_in = self.metadata_encoder(
            discrete_conditions=denoiser_input["discrete_cond_input"],
            continuous_conditions=denoiser_input["continuous_cond_input"],
        )
        diffusion_step = denoiser_input["diffusion_step"]

        x = self.init_conv(noisy_sample)
        x = self.residual_layer((x, cond_in, diffusion_step.view(B, 1)))
        y = self.final_conv(x)

        return y

    def prepare_output(self, synthesized):
        return synthesized.detach().cpu().numpy()