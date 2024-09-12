# -*- coding = utf-8 -*-
# @Time : 2024/8/6 9:18
# @Author : Yourui Han
# @File : utility.py
# @Software : PyCharm


import torch
import pandas as pd
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from TorchDiffEqPack import odesolve
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from torchdiffeq import odeint
from functools import partial
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class Arguments:
    pass


def input_args():
    arguments = Arguments()
    arguments.dataset = input("Name of the dataset. Option: embryo, etc. (default: embryo):") or "embryo"
    statepoints = input("State points of dataset (default: 1, 2, 3, 4, 5):")
    arguments.pseudo_time = [float(sp.strip())-1 for sp in statepoints.split(",")] if statepoints else [0, 1, 2, 3, 4]
    arguments.num_iters = int(input("Number of the training iterations (default: 6000):") or 4800)
    arguments.num_samples = int(input("Number of sampling points per epoch: ") or 40)
    arguments.lr = float(input("Learning rate: ") or 1.5e-3)
    arguments.GRN_num_hiddens = int(input("Number of hidden layers in GRNAutoencoder: ") or 4)
    arguments.GRN_dim_hidden = int(input("Dimension of the hidden layer in GRNAutoencoder: ") or 10)
    arguments.BRD_num_hiddens = int(input("Number of hidden layers in BRDAutoencoder: ") or 4)
    arguments.BRD_dim_hidden = int(input("Dimension of the hidden layer in BRDAutoencoder: ") or 8)
    arguments.decrease_multipleint = (input("Multiple of decrease in autoencoder: ") or 0)
    arguments.sparsity_param = (input("Sparsity parameter: ") or 0.05)
    arguments.activation = input("Activation function (default: Tanh): ") or 'Tanh'
    arguments.gpu = int(input("GPU device index (default: 0): ") or 0)
    arguments.input_dir = input("Input Files Directory (default: Input/): ") or 'Input/'
    arguments.save_dir = input("Output Files Directory (default: Output/): ") or 'Output/'
    arguments.seed = int(input("Random seed (default: 1): ") or 1)
    return arguments


class GRNAutoencoder(nn.Module):
    """
    GRNAutoencoder is a sparse autoencoder for learning the cell type-specific gene regulatory network
    that dominates cell migration to reconstruct the Waddington landscape.
    """
    def __init__(self, in_out_dim, hidden_dim, hiddens_num, activation, decrease_multipleint, sparsity_param):
        super(GRNAutoencoder, self).__init__()
        encoder_Layers_dim = [in_out_dim]
        for i in range(hiddens_num):
            encoder_Layers_dim.append(int(hidden_dim - decrease_multipleint * i))
        decoder_Layers_dim = [encoder_Layers_dim[-1]]
        decoder_Layers_dim.append(in_out_dim)
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        encoder_Layers = []
        for i in range(len(encoder_Layers_dim) - 1):
            encoder_Layers.append(nn.Linear(encoder_Layers_dim[i], encoder_Layers_dim[i + 1]))
            encoder_Layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_Layers)
        decoder_Layers = []

        decoder_Layers.append(nn.Linear(decoder_Layers_dim[-2], decoder_Layers_dim[-1]))
        self.decoder = nn.Sequential(*decoder_Layers)
        self.sparsity_param = sparsity_param
        self.sparsity_loss = torch.tensor([1.0])

    def forward(self, s, x):
        batchsize = x.shape[0]
        s = torch.tensor(s).repeat(batchsize).reshape(batchsize, 1)
        s.requires_grad = True
        state = torch.cat((s, x), dim=1)

        encoded = self.encoder(state)
        v = self.decoder(encoded)
        sparsity_param = self.sparsity_param
        eps = 1e-7
        rho_hat = torch.mean(encoded, dim=0)
        rho_hat_clamped = torch.clamp(rho_hat, min=eps, max=1. - eps)
        self.sparsity_loss = sparsity_param * torch.sum(torch.log(sparsity_param / rho_hat_clamped)) + (1 - sparsity_param) * torch.sum(
            torch.log((1 - sparsity_param) / (1 - rho_hat_clamped)))
        return v, self.sparsity_loss


class BRDAutoencoder(nn.Module):
    """
        BRDAutoencoder is a sparse autoencoder for learning the cell type-specific growth functions that drive cell
        proliferation to characterize the cell proliferation potential to reconstruct the Waddington landscape.
    """
    def __init__(self, in_out_dim, hidden_dim, hiddens_num, activation, decrease_multipleint, sparsity_param):
        super(BRDAutoencoder, self).__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        Layers_dim = [in_out_dim + 1]
        for i in range(hiddens_num-2):
            Layers_dim.append(int(hidden_dim - decrease_multipleint * i))
        Layers_dim.append(1)

        Layers = []
        for i in range(len(Layers_dim) - 2):
            Layers.append(nn.Linear(Layers_dim[i], Layers_dim[i + 1]))
            Layers.append(self.activation)
        self.encoder = nn.Sequential(*Layers)
        self.decoder = nn.Linear(Layers_dim[-2], Layers_dim[-1])
        self.sparsity_param = sparsity_param
        self.sparsity_loss = torch.tensor([1.0])

    def forward(self, s, x):
        # x is N*2
        batchsize = x.shape[0]
        s = torch.tensor(s).repeat(batchsize).reshape(batchsize, 1)
        s.requires_grad = True
        state = torch.cat((s, x), dim=1)
        encoded = self.encoder(state)
        g = self.decoder(encoded)
        sparsity_param = self.sparsity_param
        eps = 1e-7  # 防止log(0)的小正数
        rho_hat = torch.mean(encoded, dim=0)
        rho_hat_clamped = torch.clamp(rho_hat, min=eps, max=1. - eps)
        self.sparsity_loss = sparsity_param * torch.sum(torch.log(sparsity_param / rho_hat_clamped)) + (
                    1 - sparsity_param) * torch.sum(torch.log((1 - sparsity_param) / (1 - rho_hat_clamped)))
        return g, self.sparsity_loss


class RWL(nn.Module):
    """
    The particular cell development(reprogramming or differentiation) process is modeled as state continuous cellular
    dynamics and decoupled into two distinct components: cell migration and cell proliferation. And the cell migration
    is modeled as GRNAutoencoder by advection and diffusion, whereas the cell proliferation is modeled as BRDAutoencoder
    by reaction.
    """
    def __init__(self, in_out_dim, GRN_dim_hidden, GRN_num_hiddens, BRD_dim_hidden, BRD_num_hiddens, activation,
                 decrease_multipleint, sparsity_param):
        super(RWL, self).__init__()
        self.in_out_dim = in_out_dim
        self.GRN_dim_hidden = GRN_dim_hidden
        self.BRD_dim_hidden = BRD_dim_hidden
        self.sparsity_param = sparsity_param
        self.decrease_multipleint = decrease_multipleint
        self.GRN_net = GRNAutoencoder(in_out_dim, GRN_dim_hidden, GRN_num_hiddens, activation, decrease_multipleint,
                                  sparsity_param)
        self.BRD_net = BRDAutoencoder(in_out_dim, BRD_dim_hidden, BRD_num_hiddens, activation, decrease_multipleint,
                                  sparsity_param)

    def forward(self, pseudo_t, states):
        x = states[0]
        g_x = states[1]
        p = states[2]

        batchsize = x.shape[0]

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            x_gene = x[:, :x.shape[1] - 1]  # Extracting gene features
            dx_dt, _ = self.GRN_net(pseudo_t, x_gene)  # input: pseudo time and gene features
            g, _ = self.BRD_net(pseudo_t, x)  # input: pseudo time, gene features and density features
            dp_dt = g - p * trace_df_dz(dx_dt, x_gene).view(batchsize, 1)
        return (dx_dt, g, dp_dt)


def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def initialize_weights(m):
    """
    Initializing the weights of network.
    """
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def Jacobian(f, z):
    """
    Calculates Jacobian df/dz.
    """
    jac = []
    for i in range(z.shape[1]):
        df_dz = torch.autograd.grad(f[:, i], z, torch.ones_like(f[:, i]), retain_graph=True, create_graph=True)[0].view(
            z.shape[0], -1)
        jac.append(torch.unsqueeze(df_dz, 1))
    jac = torch.cat(jac, 1)
    return jac


def Gaussian_density(x, state_all, state_pt, data_train, sigma, device):
    data = data_train[state_all[state_pt]]
    sigma_matrix = sigma * torch.eye(data.shape[1]).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(data.shape[0]):
        m = torch.distributions.multivariate_normal.MultivariateNormal(data[i, :], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn / data.shape[0]
    return p_n


def Sampling_noise(num_samples, state_all, state_pt, data_train, sigma, device):
    # perturb the data with Gaussian noise
    data = data_train[state_all[state_pt]]  # data is number_sample * dimension
    sigma_matrix = sigma * torch.eye(data.shape[1])
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(data.shape[1]), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)

    if data.shape[0] < num_samples:
        samples = data[random.choices(range(0, data.shape[0]), k=num_samples)] + noise_add
    else:
        samples = data[random.sample(range(0, data.shape[0]), num_samples)]
    return samples


def loaddata(arguments, device):
    data = np.load(os.path.join(arguments.input_dir, (arguments.dataset + '.npy')), allow_pickle=True)
    data_train = []
    for i in range(data.shape[1]):
        data_train.append(torch.from_numpy(data[0, i]).type(torch.float32).to(device))
    return data_train


def growth(s, y, func, device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)
    gg = func.forward(s, y)[1]
    return (y_0, y_00, gg)


def trans_loss(s, y, func, device, odeint_step):
    outputs = func.forward(s, y)
    v = outputs[0]
    g = outputs[1]
    y_0 = torch.zeros(g.shape).type(torch.float32).to(device)
    y_00 = torch.zeros(v.shape).type(torch.float32).to(device)
    g_growth = partial(growth, func=func, device=device)
    if torch.is_nonzero(s):
        _, _, g_int = odeint(g_growth, (y_00, y_0, y_0), torch.tensor([0, s]).type(torch.float32).to(device), atol=1e-5,
                             rtol=1e-5, method='midpoint', options={'step_size': odeint_step})
        f_int = (torch.norm(v, dim=1) ** 2).unsqueeze(1) * g_int[-1] + (torch.norm(g, dim=1) ** 2).unsqueeze(1) / g_int[-1]
        return (y_00, y_0, f_int)
    else:
        return (y_00, y_0, y_0)


def train_model(mse, func, arguments, data_train, train_time, integral_time, sigma_now, options, device, itr):
    global global_variable
    warnings.filterwarnings("ignore")

    loss = 0
    loss_x = 0
    odeint_step = 1.0
    L2_value1 = torch.zeros(1, len(data_train) - 1).type(torch.float32).to(device)
    L2_value2 = torch.zeros(1, len(data_train) - 1).type(torch.float32).to(device)

    for i in range(len(train_time) - 1):
        x = Sampling_noise(arguments.num_samples, train_time, i + 1, data_train, 0.01, device)
        x.requires_grad = True
        p_diff_s1 = torch.ones(x.shape[0], 1).type(torch.float32).to(device)
        g_s1 = p_diff_s1
        options.update({'t0': integral_time[i + 1]})
        options.update({'t1': integral_time[0]})
        z_s0, g_s0, p_diff_s0 = odesolve(func, y0=(x, g_s1, p_diff_s1), options=options)
        p_rl = Gaussian_density(z_s0, train_time, 0, data_train, sigma_now, device)  # normalized density

        zero_den = (p_rl < 1e-16).nonzero(as_tuple=True)[0]
        p_rl[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        p_rl = p_rl / torch.exp(p_diff_s0.view(-1))
        p = Gaussian_density(x, train_time, i + 1, data_train, sigma_now, device) * torch.tensor(
            data_train[i + 1].shape[0] / data_train[0].shape[0])  # mass

        L2_value1[0][i] = mse(p, p_rl)

        true_sample = data_train[train_time[0]]
        loss_x = mse(torch.mean(true_sample[:, :true_sample.shape[1] - 1], dim=0),
                     torch.mean(z_s0[:, :z_s0.shape[1] - 1], dim=0))

        loss = loss + L2_value1[0][i] * 1e4
        loss = loss + loss_x * 1e-2
        # loss between each two time points
        options.update({'t0': integral_time[i + 1]})
        options.update({'t1': integral_time[i]})
        z_s0, g_s0, p_diff_s0 = odesolve(func, y0=(x, g_s1, p_diff_s1), options=options)

        p_rs = Gaussian_density(z_s0, train_time, i, data_train, sigma_now, device) * torch.tensor(
            data_train[i].shape[0] / data_train[0].shape[0])

        # find zero density
        zero_den = (p_rs < 1e-16).nonzero(as_tuple=True)[0]
        p_rs[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        p_rs = p_rs / torch.exp(p_diff_s0.view(-1))

        L2_value2[0][i] = mse(p, p_rs)
        loss = loss + L2_value2[0][i] * 1e4

        true_sample = data_train[train_time[i]]
        loss_x = mse(torch.mean(true_sample[:, :true_sample.shape[1] - 1], dim=0),
                     torch.mean(z_s0[:, :z_s0.shape[1] - 1], dim=0))
        loss = loss + loss_x * 1e-2
        sparsity_loss1 = func.GRN_net.sparsity_loss
        sparsity_loss2 = func.BRD_net.sparsity_loss
        loss = loss + sparsity_loss1 + sparsity_loss2
    # compute transport cost efficiency
    transport_cost = partial(trans_loss, func=func, device=device, odeint_step=odeint_step)
    x0 = Sampling_noise(arguments.num_samples, train_time, 0, data_train, 0.02, device)
    p_diff_s00 = torch.zeros(x0.shape[0], 1).type(torch.float32).to(device)
    g_s00 = p_diff_s00
    _, _, loss1 = odeint(transport_cost, y0=(x0, g_s00, p_diff_s00),
                         t=torch.tensor([0, integral_time[-1]]).type(torch.float32).to(device), atol=1e-5, rtol=1e-5,
                         method='midpoint', options={'step_size': odeint_step})
    loss = loss + integral_time[-1] * loss1[-1].mean(0)

    if itr > 1:
        if ((itr % 100 == 0) and (itr <= arguments.num_iters - 400) and (sigma_now > 0.02) and (L2_value1.mean() <= 0.0003)):
            sigma_now = sigma_now / 2

    return loss, loss1, sigma_now, L2_value1, L2_value2


