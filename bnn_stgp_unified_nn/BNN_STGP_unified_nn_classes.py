import os
import sys
import time
import math
import copy
import random
import warnings
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

warnings.filterwarnings("ignore")

from GP_comp.GP import gp_eigen_value, gp_eigen_funcs_fast, generate_grids, gp_std_grids
from simulate_data import simulate_data_circle_square
from utils import generate_linear_data, plot_mse, plot_sigma_squared, plot_image

from utils import sample_inverse_gamma

class Multimodal_BNNSTGP(nn.Module):
    def __init__(self, input_dim, n_hid_list, output_dim, w_dim, n_knots,
                 phi_cr_list, phi_cw_list, lambda_cr_list, lambda_cw_list, act='relu'):
        super(Multimodal_BNNSTGP, self).__init__()

        self.input_dim = input_dim
        self.n_hid_list = n_hid_list
        self.output_dim = output_dim
        self.w_dim = w_dim
        self.n_knots = n_knots
        self.num_modalities = len(phi_cr_list)
        self.phi_cr_names = []
        self.phi_cw_names = []
        self.lambda_cr_names = []
        self.lambda_cw_names = []

        for m in range(self.num_modalities):
            phi_cr = phi_cr_list[m] if torch.is_tensor(phi_cr_list[m]) else torch.tensor(phi_cr_list[m], dtype=torch.float32)
            phi_cw = phi_cw_list[m] if torch.is_tensor(phi_cw_list[m]) else torch.tensor(phi_cw_list[m], dtype=torch.float32)
            lambda_cr = torch.as_tensor(lambda_cr_list[m], dtype=torch.float32, device=phi_cr.device)
            lambda_cw = torch.as_tensor(lambda_cw_list[m], dtype=torch.float32, device=phi_cw.device)

            phi_cr_name = f"phi_cr_{m}"
            phi_cw_name = f"phi_cw_{m}"
            lambda_cr_name = f"lambda_cr_{m}"
            lambda_cw_name = f"lambda_cw_{m}"
            self.register_buffer(phi_cr_name, phi_cr)
            self.register_buffer(phi_cw_name, phi_cw)
            self.register_buffer(lambda_cr_name, lambda_cr)
            self.register_buffer(lambda_cw_name, lambda_cw)
            self.phi_cr_names.append(phi_cr_name)
            self.phi_cw_names.append(phi_cw_name)
            self.lambda_cr_names.append(lambda_cr_name)
            self.lambda_cw_names.append(lambda_cw_name)

        self.sigma_lambda_r_squared_list = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0), requires_grad=False)
            for _ in range(self.num_modalities)
        ])
        self.sigma_lambda_w_squared_list = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0), requires_grad=False)
            for _ in range(self.num_modalities)
        ])

        self.c_r_list = nn.ParameterList()
        self.c_w_list = nn.ParameterList()
        self.TOTAL_ENTRIES_OF_CR = []
        self.TOTAL_ENTRIES_OF_CW = []
        for m in range(self.num_modalities):
            lambda_cr = self._lambda_cr(m)
            lambda_cw = self._lambda_cw(m)
            c_r = nn.Parameter(torch.randn(n_knots, device=lambda_cr.device) * lambda_cr)
            c_w = nn.Parameter(torch.randn(n_knots, n_hid_list[0], device=lambda_cw.device) * lambda_cw.unsqueeze(1))
            self.c_r_list.append(c_r)
            self.c_w_list.append(c_w)
            self.TOTAL_ENTRIES_OF_CR.append(c_r.numel())
            self.TOTAL_ENTRIES_OF_CW.append(c_w.numel())

        
        value = sample_inverse_gamma(2.0, 1.0, size=1).squeeze()
        # However, Guoxuan said the initial value of sigma_squared can be set at 1.0 for now.
        self.sigma_theta_squared = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        value = sample_inverse_gamma(2.0, 1.0, size=1).squeeze()
        # However, Guoxuan said the initial value of sigma_squared can be set at 1.0 for now.
        self.sigma_squared = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Invalid activation function %s' % act)

        #create one flat parameter so all other operations stay the same
        total_params = 0
        current_size = n_hid_list[0]
        for i in range(1, len(n_hid_list)):
            total_params += current_size * n_hid_list[i] + n_hid_list[i]
            current_size = n_hid_list[i]
        total_params += current_size * output_dim + output_dim
        self.fully_connected_nn = nn.Parameter(torch.Tensor(total_params).normal_(0, math.sqrt(self.sigma_theta_squared.item())))

        # store architecture info
        self.fc_arch = n_hid_list.copy()
        self.fc_arch.append(output_dim)

        # helper lists for forward pass
        self._fc_weights = []
        self._fc_biases = []
        self._setup_fc_layers()
        
        self.ksi = nn.Parameter(torch.Tensor(n_hid_list[0]).normal_(0, math.sqrt(self.sigma_theta_squared.item())))
        self.NUM_PARAMS = self.count_num_params()

    def _phi_cr(self, m):
        return getattr(self, self.phi_cr_names[m])

    def _phi_cw(self, m):
        return getattr(self, self.phi_cw_names[m])

    def _lambda_cr(self, m):
        return getattr(self, self.lambda_cr_names[m])

    def _lambda_cw(self, m):
        return getattr(self, self.lambda_cw_names[m])

    def _setup_fc_layers(self):
        #purpose of this function is to reshape fully_connected_nn into multiple layers for forward pass
        self._fc_weights = []
        self._fc_biases = []
        idx = 0
        input_size = self.n_hid_list[0]
        
        for i, output_size in enumerate(self.fc_arch[1:]):
            weight_size = input_size * output_size
            weight = self.fully_connected_nn[idx:idx + weight_size].reshape(input_size, output_size)
            self._fc_weights.append(weight)
            idx += weight_size
            
            bias = self.fully_connected_nn[idx:idx + output_size]
            self._fc_biases.append(bias)
            idx += output_size
            
            input_size = output_size

    def input_layer_beta_r(self, m):
        beta_r = torch.mv(self._phi_cr(m), self.c_r_list[m])  # (V,)
        beta_r = torch.sqrt(self.sigma_lambda_r_squared_list[m]) * beta_r
        return beta_r

    def input_layer_beta_w(self, m):
        beta_w = torch.mm(self._phi_cw(m), self.c_w_list[m])  # (V, U1)
        beta_w = torch.sqrt(self.sigma_lambda_w_squared_list[m]) * beta_w
        return beta_w

    def input_layer_output(self, x, m, mode="train"):
        beta_r = self.input_layer_beta_r(m)

        if mode == "train":
            k = 50.0  # steepness hyperparam (tune)
            mask = 0.5 * (1.0 + torch.tanh(k * beta_r))
        elif mode == "inference":
            mask = (beta_r > 0).float()
        else:
            raise ValueError('mode can only be one of train or inference')

        beta_w = self.input_layer_beta_w(m)
        beta_eff = beta_w * mask.unsqueeze(1)  # (V, U1)
        return torch.mm(x, beta_eff)  # (batch, U1)

    def forward(self, x_list, w, mode = "train"):
        # Rebuild views from the current parameter tensor so they stay on the
        # same device as fully_connected_nn after model.to(device).
        self._setup_fc_layers()
        # ----- Region selection GP: beta_r(s) in R^V
        out = sum(
            self.input_layer_output(x_list[m], m, mode=mode)
            for m in range(self.num_modalities)
        ) + self.ksi
        out = self.act(out)
        for i, (weight, bias) in enumerate(zip(self._fc_weights, self._fc_biases)):
            #multiply each layer and add bias one by one
            out = torch.mm(out, weight) + bias
            if i < len(self._fc_weights) - 1:
                out = self.act(out)
        return out
    def log_prior(self):
        logprior = 0.0
        for m in range(self.num_modalities):
            logprior += 0.5 * ((self.c_r_list[m] ** 2) / self._lambda_cr(m)).sum() / self.sigma_lambda_r_squared_list[m]
            logprior += 0.5 * ((self.c_w_list[m] ** 2) / self._lambda_cw(m).unsqueeze(1)).sum() / self.sigma_lambda_w_squared_list[m]
        logprior += 0.5 * (self.fully_connected_nn ** 2).sum() / self.sigma_theta_squared
        logprior += 0.5 * (self.ksi ** 2).sum() / self.sigma_theta_squared
        return logprior

    def count_num_params(self):
        count = 0
        count += self.ksi.numel()
        count += self.fully_connected_nn.numel() 
        return count
        
    def sample_and_set_sigma_squared(self, n, loss, a=1000, b=1.0):
        with torch.no_grad():
            new_a = a + n / 2.0
            new_b = b + loss / 2.0
            sigma_squared = sample_inverse_gamma(
                new_a, new_b, size=1, device=self.sigma_squared.device
            ).squeeze()
            self.sigma_squared.copy_(sigma_squared)
            return self.sigma_squared

    def sample_and_set_sigma_lambda_r_squared(self, a_lambda_r=2.0, b_lambda_r=1.0):
        with torch.no_grad():
            values = []
            for m in range(self.num_modalities):
                new_a = a_lambda_r + self.TOTAL_ENTRIES_OF_CR[m] / 2.0
                new_b = b_lambda_r + 0.5 * ((self.c_r_list[m] ** 2) / self._lambda_cr(m)).sum()
                val = sample_inverse_gamma(
                    new_a, new_b, size=1, device=self.sigma_lambda_r_squared_list[m].device
                ).squeeze()
                self.sigma_lambda_r_squared_list[m].copy_(val)
                values.append(self.sigma_lambda_r_squared_list[m])
            return values
    
    def sample_and_set_sigma_lambda_w_squared(self, a_lambda_w=2.0, b_lambda_w=1.0):
        with torch.no_grad():
            values = []
            for m in range(self.num_modalities):
                new_a = a_lambda_w + self.TOTAL_ENTRIES_OF_CW[m] / 2.0
                new_b = b_lambda_w + 0.5 * ((self.c_w_list[m] ** 2) / self._lambda_cw(m).unsqueeze(1)).sum()
                val = sample_inverse_gamma(
                    new_a, new_b, size=1, device=self.sigma_lambda_w_squared_list[m].device
                ).squeeze()
                self.sigma_lambda_w_squared_list[m].copy_(val)
                values.append(self.sigma_lambda_w_squared_list[m])
            return values

    def sample_and_set_sigma_theta_squared(self, a_theta=2.0, b_theta=1.0):
        """
        Sample σβ^2 from an Inverse-Gamma distribution using equation (21), with data dependency via β^T β.
        Returns:
            torch.Tensor: Sampled σβ^2 from the Inverse-Gamma distribution.
        """
        with torch.no_grad():
            new_a_theta = a_theta + self.NUM_PARAMS / 2.0
            new_b_theta = b_theta + torch.sum(self.fully_connected_nn ** 2) / 2.0 + torch.sum(self.ksi ** 2) / 2.0

            sigma_theta_squared = sample_inverse_gamma(
                new_a_theta, new_b_theta, size=1, device=self.sigma_theta_squared.device
            ).squeeze()
            self.sigma_theta_squared.copy_(sigma_theta_squared)
            return self.sigma_theta_squared

    def step_approximation(self, x, steep):
        return (1 / 2 + 1 / np.pi * torch.arctan(steep * x))

class SGLD(Optimizer):
    def __init__(self, params, lr = required, langevin = True):
        self.langevin = langevin
        defaults = dict(lr=lr)
        super(SGLD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        loss = None
        
        for group in self.param_groups:
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                
                if self.langevin == True:
                    langevin_noise = p.new(p.size()).normal_(mean=0, std=1)/np.sqrt(group['lr'])
                    p.add_(0.5*d_p + langevin_noise, alpha = -group['lr'])

                else:
                    p.add_(0.5*d_p, alpha = -group['lr'])


        return loss

class Net(object):

    def __init__(self, task='binary', lr=1e-3, input_dim=784, n_hid_list=[32,32], output_dim=1, w_dim=1, n_knots=66,
                 N_train=200, phi_cr_list=None, lambda_cr_list=None, phi_cw_list=None, lambda_cw_list=None,
                 langevin=True, step_decay_epoch=100, step_gamma=0.1, act='relu'):
        
        #print(' Creating Net!! ')
        self.task = task
        if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError('Invalid task %s' % task)
        self.lr = lr

        self.input_dim = input_dim
        self.n_hid_list = n_hid_list
        self.output_dim = output_dim
        self.w_dim = w_dim
        
        self.n_knots = n_knots
        self.phi_cr_list = phi_cr_list
        self.lambda_cr_list = lambda_cr_list
        self.phi_cw_list = phi_cw_list
        self.lambda_cw_list = lambda_cw_list
        self.num_modalities = len(self.phi_cr_list)
        self.act = act

        self.N_train = N_train
        self.langevin = langevin
        self.step_decay_epoch = step_decay_epoch
        self.step_gamma = step_gamma

        self.create_net()
        self.create_opt()
        self.epoch = 0
        
        self.weight_set_samples = []
        self.r2_train = []
        self.r2_test = []
        self.saved_r2_train = []
        self.saved_r2_test = []

    def create_net(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Multimodal_BNNSTGP(
            input_dim=self.input_dim,
            n_hid_list=self.n_hid_list,
            output_dim=self.output_dim,
            w_dim=self.w_dim,
            n_knots=self.n_knots,
            phi_cr_list=[torch.tensor(phi, dtype=torch.float32) for phi in self.phi_cr_list],
            phi_cw_list=[torch.tensor(phi, dtype=torch.float32) for phi in self.phi_cw_list],
            lambda_cr_list=self.lambda_cr_list,
            lambda_cw_list=self.lambda_cw_list,
            act=self.act
        )
        self.model.to(self.device)


    def create_opt(self):
        self.optimizer = SGLD(params=self.model.parameters(), lr=self.lr, langevin = self.langevin)
        self.scheduler = StepLR(self.optimizer, step_size = self.step_decay_epoch, gamma=self.step_gamma)


    def fit(self, x_list, w, y, threshold=0.5):
        x_list = [x.to(self.device) for x in x_list]
        w = w.to(self.device)
        y = y.float().to(self.device).reshape(-1, 1)
        self.optimizer.zero_grad()

        out = self.model(x_list, w, mode="train")
        mse_loss = F.mse_loss(out, y, reduction='mean')
        self.r2_train.append(1 - mse_loss.detach().item() / torch.var(y).item())
        adjusted_loss = mse_loss * self.N_train / self.model.sigma_squared / 2.0
        total_loss = adjusted_loss + self.model.log_prior()
        
        total_loss.backward()
        self.optimizer.step() 

        pred = 0.0
        accu = 0.0
        return mse_loss, accu 

    def get_nb_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


    def save_net_weights(self, max_samples):
        
        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)
            self.saved_r2_train.pop(0) 
            self.saved_r2_test.pop(0)
            
        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))
        self.saved_r2_train.append(self.r2_train[-1] if self.r2_train else np.nan)
        self.saved_r2_test.append(self.r2_test[-1] if self.r2_test else np.nan)
        #print(' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples) )


    def all_sample_eval(self, x_list, w, y, threshold=0.5):
        with torch.no_grad():
            x_list = [x.to(self.device) for x in x_list]
            w = w.to(self.device)
            y = y.float().to(self.device).reshape(-1, 1)
            self.optimizer.zero_grad()

            out = self.model(x_list, w, mode="inference")
            mse_loss = F.mse_loss(out, y, reduction='mean')
            self.r2_test.append(1 - mse_loss.detach().item() / torch.var(y).item())

            pred = 0.0
            accu = 0.0
            return mse_loss, accu 


    def save(self, filename):
        print('Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler}, filename)

    def load(self, filename):
        print('Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        self.scheduler = state_dict['scheduler']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch

class mydata(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, i):
        x_out = [x_mod[i].reshape(-1) for x_mod in self.x_list]
        w = torch.tensor([1.])
        y = torch.tensor([self.y_list[i]])
        return (x_out, w), y

def create_dataloaders(X_list, y, val_ratio=0.2, batch_size=32, shuffle=True):
    """
    Splits X_list, y into training and validation DataLoaders.

    Args:
        X_list (list[torch.Tensor]): List of feature tensors, each of shape (n_samples, ...).
        y (torch.Tensor): Target tensor, shape (n_samples,).
        val_ratio (float): Proportion of dataset to use for validation.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the training DataLoader.

    Returns:
        train_loader, val_loader, n_train, n_val
    """
    dataset = mydata(X_list, y)

    n_samples = len(dataset)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, n_train, n_val

class MNIST_runner:
    def __init__(self, train_loader, val_loader, lr=5e-7, a_for_eigen_cr_list=[0.0001, 0.0001],
                 b_for_eigen_cr_list=[200, 200], a_for_eigen_cw_list=[0.0001, 0.0001], 
                 b_for_eigen_cw_list=[200, 200], a_for_sigma_squared=1000, langevin=True, 
                 seed=17, act='relu', in_feature_list=[100, 200], hidden_unit_list=[32,32], step_gamma=0.5, step_decay_epoch=50, poly_degree=30):
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.lr = lr
        self.langevin = langevin
        self.seed = seed
        self.act = act
        self.curr_epoch = 0
        self.total_epochs = 0
        
        self.start_save = 0
        self.save_every = 2
        self.N_saves = 100
        self.test_every = 1
        self.print_every = 10
        
        self.loss_train = []
        self.accu_train = []
        self.loss_val = []
        self.accu_val = []

        self.n_hid_list = hidden_unit_list
        self.in_feature_list = in_feature_list
        self.num_modalities = len(in_feature_list)

        self.selected_voxel_prop_train = [[] for _ in range(self.num_modalities)]
        self.selected_voxel_prop_val = [[] for _ in range(self.num_modalities)]
        self.a_for_sigma_squared = a_for_sigma_squared

        phi_cr_list = []
        phi_cw_list = []

        for m, in_feature in enumerate(self.in_feature_list):
            grids_cr = generate_grids(dimensions=2, num_grids=in_feature)
            grids_cr, _, _ = gp_std_grids(grids_cr)
            phi_cr = gp_eigen_funcs_fast(
                grids_cr,
                poly_degree=poly_degree,
                a=a_for_eigen_cr_list[m],
                b=b_for_eigen_cr_list[m]
            )
            phi_cr_list.append(np.array(phi_cr))
        
            grids_cw = generate_grids(dimensions=2, num_grids=in_feature)
            phi_cw = gp_eigen_funcs_fast(
                grids_cw,
                poly_degree=poly_degree,
                a=a_for_eigen_cw_list[m],
                b=b_for_eigen_cw_list[m]
            )
            phi_cw_list.append(np.array(phi_cw))

        lambda_np_cr_list = []
        lambda_np_cw_list = []
        
        for m in range(self.num_modalities):
            lambda_np_cr = gp_eigen_value(
                poly_degree=poly_degree,
                a=a_for_eigen_cr_list[m],
                b=b_for_eigen_cr_list[m]
            )
            lambda_np_cr_list.append(lambda_np_cr)
        
            lambda_np_cw = gp_eigen_value(
                poly_degree=poly_degree,
                a=a_for_eigen_cw_list[m],
                b=b_for_eigen_cw_list[m]
            )
            lambda_np_cw_list.append(lambda_np_cw)
        
        torch.set_default_dtype(torch.float32)

        self.net = Net(
            lr=lr,
            input_dim=self.in_feature_list[0] * self.in_feature_list[0],
            n_hid_list=self.n_hid_list,
            output_dim=1,
            w_dim=1,
            n_knots=phi_cr_list[0].shape[1],
            N_train=len(self.train_loader.dataset.indices),
            phi_cr_list=phi_cr_list,
            lambda_cr_list=lambda_np_cr_list,
            phi_cw_list=phi_cw_list,
            lambda_cw_list=lambda_np_cw_list,
            langevin=langevin,
            step_decay_epoch=step_decay_epoch,
            step_gamma=step_gamma,
            act=act,
            task='regression'
        )
        
    def _get_saved_sample_start_idx(self, burnin_ratio=0.5):
        if not 0 <= burnin_ratio < 1:
            raise ValueError("burnin_ratio must be in [0, 1).")

        n_saved = len(self.net.weight_set_samples)
        if n_saved == 0:
            raise ValueError("No saved posterior samples are available.")

        return int(n_saved * burnin_ratio)

    def get_saved_posterior_summary(self, burnin_ratio=0.5):
        start_idx = self._get_saved_sample_start_idx(burnin_ratio)
        return {
            "start_idx": start_idx,
            "weight_samples": self.net.weight_set_samples[start_idx:],
            "r2_train": np.asarray(self.net.saved_r2_train[start_idx:], dtype=float),
            "r2_test": np.asarray(self.net.saved_r2_test[start_idx:], dtype=float)
        }

    def _mean_saved_state_dict(self, weight_samples):
        # iterate through all samples and take mean of all parameters
        mean_state_dict = {}

        for key in weight_samples[0].keys():
            first_value = weight_samples[0][key]
            if torch.is_floating_point(first_value) or torch.is_complex(first_value):
                stacked_values = torch.stack([
                    weight_dict[key].detach().to(self.net.device)
                    for weight_dict in weight_samples
                ], dim=0)
                mean_state_dict[key] = stacked_values.mean(dim=0)
            else:
                mean_state_dict[key] = first_value.detach().clone().to(self.net.device)

        return mean_state_dict

    def _get_split_tensors(self, data_loader):
        # put data from data loader into 1 list for each x and y
        split_dataset = data_loader.dataset
        full_dataset = split_dataset.dataset
        indices = split_dataset.indices

        x_list = [
            full_dataset.x_list[m][indices, :].to(self.net.device)
            for m in range(self.num_modalities)
        ]
        y = full_dataset.y_list[indices].float().to(self.net.device).reshape(-1, 1)
        return x_list, y

    def _calculate_r2_from_current_model(self, data_loader, mode):
        #get r2 from current model
        x_list, y = self._get_split_tensors(data_loader)
        pred = self.net.model(
            x_list,
            torch.tensor([1.0], device=self.net.device),
            mode=mode
        ).detach().reshape(-1, 1)

        mse_loss = F.mse_loss(pred, y, reduction='mean')
        y_var = torch.var(y)
        r2 = 1 - mse_loss.detach().item() / y_var.item()
        return r2

    def calculate_r2_from_mean_parameters(self, burnin_ratio=0.5):
        # make copy of current state of model, get the mean samples, load mean state, calculate r2, restore.  
        posterior_summary = self.get_saved_posterior_summary(burnin_ratio)
        weight_samples = posterior_summary["weight_samples"]
        current_state_dict = copy.deepcopy(self.net.model.state_dict())
        mean_state_dict = self._mean_saved_state_dict(weight_samples)

        with torch.no_grad():
            self.net.model.load_state_dict(mean_state_dict)
            r2_train = self._calculate_r2_from_current_model(self.train_loader, mode="train")
            r2_test = self._calculate_r2_from_current_model(self.val_loader, mode="inference")
            self.net.model.load_state_dict(current_state_dict)

        return {"r2_train": r2_train,"r2_test": r2_test,}

    def calculate_beta(self, burnin_ratio=0.5):
        beta_r_samples_list = [[] for _ in range(self.num_modalities)]
        posterior_summary = self.get_saved_posterior_summary(burnin_ratio)

        for weight_dict in posterior_summary["weight_samples"]:
            with torch.no_grad():
                self.net.model.load_state_dict(weight_dict)

                for m in range(self.net.model.num_modalities):
                    beta_r_m = self.net.model.input_layer_beta_r(m)
                    beta_r_samples_list[m].append(beta_r_m.detach().cpu().numpy())

        beta_r_arr_list = [np.stack(beta_samples, axis=0) for beta_samples in beta_r_samples_list]
        return beta_r_arr_list

    def calculate_mask(self, beta, gamma=0.5):
        T, V = beta.shape # t = number of posterior samples, v = num of voxels
        
        p_hat = (beta > 0).mean(axis=0)   # collapses the num_samples, true if positive, avergaes across all samples
        order = np.argsort(-p_hat)
        p_sorted = p_hat[order]
        fdr = np.cumsum(1 - p_sorted) / np.arange(1, len(p_sorted) + 1)
        valid = np.where(fdr <= gamma)[0]
        if valid.size > 0:
            r = int(valid[-1] + 1)
            delta = float(p_sorted[r - 1])
        else:
            r, delta = 0, 1.0
        mask_final = p_hat > delta
        return mask_final, p_hat, delta
    
    def train(self, n_epochs):
        self.total_epochs += n_epochs
        for i in range(self.curr_epoch, self.total_epochs):
            tic = time.time()
            self.net.scheduler.step()
            
            for (x_list, w), y in self.train_loader:
                mse_loss, accu = self.net.fit(x_list, w, y)
                self.net.model.sample_and_set_sigma_lambda_w_squared()
                self.net.model.sample_and_set_sigma_lambda_r_squared()
                self.net.model.sample_and_set_sigma_theta_squared()
                #self.net.model.sample_and_set_sigma_squared(len(train_loader.dataset), mse_loss)
                
            with torch.no_grad():
                X_train_list = [
                    self.train_loader.dataset.dataset.x_list[m][self.train_loader.dataset.indices, :].to(self.net.device)
                    for m in range(self.num_modalities)
                ]
                y_train = self.train_loader.dataset.dataset.y_list[self.train_loader.dataset.indices].to(self.net.device)
                # pred = self.net.model(X1_train, X2_train, 1).to("cpu").detach().numpy()
                # sse = np.sum((y_train.numpy() - pred.reshape(-1)) ** 2)
                pred = self.net.model(X_train_list, torch.tensor([1.0], device=self.net.device), mode="train").detach()
                sse = torch.sum((y_train - pred.view(-1)) ** 2)
                self.net.model.sample_and_set_sigma_squared(len(self.train_loader.dataset), sse, a=self.a_for_sigma_squared)
                self.loss_train.append(sse.item() / (y_train.shape[0]))
                self.accu_train.append(accu)
            # train_props = self.compute_selected_voxel_proportions(c=1.0)
            # for m, prop in enumerate(train_props):
            #     self.selected_voxel_prop_train[m].append(prop)


            toc = time.time()
    
            if i % self.test_every == 0:
                with torch.no_grad():
                    for (x_list_val, w), y_val in self.val_loader:
                        mse_loss, accu = self.net.all_sample_eval(x_list_val, torch.tensor(1), y_val)
                    tic = time.time()
                    X_val_list = [
                        self.val_loader.dataset.dataset.x_list[m][self.val_loader.dataset.indices, :].to(self.net.device)
                        for m in range(self.num_modalities)
                    ]
                    y_val = self.val_loader.dataset.dataset.y_list[self.val_loader.dataset.indices].to(self.net.device)

                    pred = self.net.model(X_val_list, torch.tensor([1.0], device=self.net.device), mode="inference").detach()
                    sse = torch.sum((y_val - pred.view(-1)) ** 2)
                    # pred = self.net.model(X1_val, X2_val, 1).to("cpu").detach().numpy()
                    # sse = np.sum((y_val.numpy() - pred.reshape(-1)) ** 2)
                    self.loss_val.append(sse.item() / (y_val.shape[0]))
                    self.accu_val.append(accu)
    
                    toc = time.time()

            if i > self.start_save and i % self.save_every == 0:
                self.net.save_net_weights(max_samples = self.N_saves)

            if i % self.print_every == 0:
                print('Epoch %d, train accuracy %.2f%%, loss_train=%s' % (i, self.accu_train[-1]*100, self.loss_train[-1]))
                print('Epoch %d, test accuracy %.2f%%, loss_val=%s' % (i, self.accu_val[-1]*100, self.loss_val[-1]))
                # for m, prop in enumerate(train_props):
                #     print(f"Epoch {i}, modality {m+1}, proportion selected = {prop:.4f}")
            self.curr_epoch += n_epochs
        
    def compute_selected_voxel_proportions(self, c=1.0):
        props = []
        with torch.no_grad():
            for m in range(self.net.model.num_modalities):
                beta_r_m = self.net.model.input_layer_beta_r(m)
    
                prop_m = (torch.abs(beta_r_m) > c).float().mean().item()
                props.append(prop_m)
    
                print(
                    f"modality {m+1}: "
                    f"mean|beta_r|={torch.abs(beta_r_m).mean().item():.4f}, "
                    f"max|beta_r|={torch.abs(beta_r_m).max().item():.4f}, "
                    f"prop(|beta_r|>{c})={prop_m:.4f}"
                )
        return props

class Experiment:
    def __init__(self):
        self.trainers = []
        self.results = []
        self.true_betas = []

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_npz_data(self, sample_size, file_path="../circle_square_data.npz"):
        data = np.load(file_path, allow_pickle=True)
    
        img1 = data["img1"][:sample_size]
        img2 = data["img2"][:sample_size]
        y = data["y"][:sample_size]
    
        x_list = [
            torch.from_numpy(img1).float().to("cpu"),
            torch.from_numpy(img2).float().to("cpu"),
        ]
        y = torch.from_numpy(y).float().to("cpu")
    
        true_beta_list = [
            data["true_beta1"],
            data["true_beta2"],
        ]
    
        return x_list, y, true_beta_list


    def load_simulation_data(self, sample_size=1000, r2=0.8, in_feature_list=[100, 200], seed=42):
        n = sample_size
        v_list1, v_list2, true_beta1, true_beta2, img1, img2, Y = simulate_data_circle_square(
            n=n,
            r2=r2,
            dim1=in_feature_list[0],
            dim2=in_feature_list[1],
            random_seed=seed
        )

        x_list = [
            torch.from_numpy(img1).float().to('cpu'),
            torch.from_numpy(img2).float().to('cpu')
        ]
        y = torch.from_numpy(Y).float().to('cpu')
        true_beta_list = [true_beta1, true_beta2]

        return x_list, y, true_beta_list
        
    def print_confusion_metrics(self, cm):
        TN, FP, FN, TP = cm.ravel()
        selection_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        true_positive_rate = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        false_discovery_rate = FP / (FP + TP) if (FP + TP) > 0 else 0.0
        print(f"Selection Accuracy:     {selection_accuracy:.4f}")
        print(f"True Positive Rate:     {true_positive_rate:.4f}")
        print(f"False Positive Rate:    {false_positive_rate:.4f}")
        print(f"False Discovery Rate:   {false_discovery_rate:.4f}")

    def run_whole_experiment(self, sample_size, data_type='npz', seed=42, true_r2=0.8,
                             gamma=0.1, val_ratio=0.5, a_for_eigen_cr_list=[0.0001, 0.0001],
                             b_for_eigen_cr_list=[200, 200], a_for_eigen_cw_list=[0.0001, 0.0001],
                             b_for_eigen_cw_list=[200, 200], a_for_sigma_squared=1000, in_feature_list=[100, 200],
                             hidden_unit_list=[32,32], step_gamma=0.5, step_decay_epoch=50, lr=1e-6, epochs=200, poly_degree=30):
        self.set_seed(seed)
        #data_type = npz/simulation
        r2 = true_r2          
        if data_type == 'npz':
            X_list, y, true_beta_list = self.load_npz_data(sample_size)
        else:
            X_list, y, true_beta_list = self.load_simulation_data(
                sample_size=sample_size,
                r2=r2,
                in_feature_list=in_feature_list,
                seed=seed
            )
        #y = scale_data(y, torch.mean(y), torch.std(y))
        train_loader, val_loader, n_train, n_test = create_dataloaders(X_list, y, val_ratio=val_ratio, batch_size=64)
        trainer = MNIST_runner(
            train_loader,
            val_loader,
            lr=lr,
            a_for_eigen_cr_list=a_for_eigen_cr_list,
            b_for_eigen_cr_list=b_for_eigen_cr_list,
            a_for_eigen_cw_list=a_for_eigen_cw_list,
            b_for_eigen_cw_list=b_for_eigen_cw_list,
            a_for_sigma_squared=a_for_sigma_squared,
            act="relu",
            in_feature_list=in_feature_list,
            hidden_unit_list=hidden_unit_list,
            step_gamma=step_gamma,
            step_decay_epoch=step_decay_epoch,
            poly_degree=poly_degree
        )
        trainer.train(epochs)


        self.trainers.append(trainer)
        self.true_betas.append(true_beta_list)
        return trainer
        
    def process_results(self, trainer, burnin_ratio=0.5, gamma=0.1, true_beta_list=None, sample_size=None):
        if sample_size is None:
            sample_size = len(trainer.train_loader.dataset) + len(trainer.val_loader.dataset)

        if true_beta_list is None and trainer in self.trainers:
            trainer_idx = self.trainers.index(trainer)
            if trainer_idx < len(self.true_betas):
                true_beta_list = self.true_betas[trainer_idx]

        posterior_summary = trainer.get_saved_posterior_summary(burnin_ratio)
        mean_parameter_r2 = trainer.calculate_r2_from_mean_parameters(burnin_ratio)
        print(f"Sample Size: {sample_size}")
        print(f"Average saved r2_train:{np.nanmean(posterior_summary['r2_train'])}")
        print(f"Average saved r2_test:{np.nanmean(posterior_summary['r2_test'])}")
        print(f"Mean-parameter r2_train:{mean_parameter_r2['r2_train']}")
        print(f"Mean-parameter r2_test:{mean_parameter_r2['r2_test']}")
        
        beta_list = trainer.calculate_beta(burnin_ratio)

        mask_list = []
        p_hat_list = []
        delta_list = []
        confusion_matrix_list = []

        for m, beta in enumerate(beta_list):
            mask, p_hat, delta = trainer.calculate_mask(beta, gamma=gamma)
            mask_list.append(mask)
            p_hat_list.append(p_hat)
            delta_list.append(delta)
            if true_beta_list is not None:
                true_selection = np.asarray(true_beta_list[m]).astype(bool).astype(int).ravel()
                predicted_selection = np.asarray(mask).astype(bool).astype(int).ravel()
                confusion_matrix_list.append(
                    confusion_matrix(true_selection, predicted_selection, labels=[0, 1])
                )

        run_result = {
            "beta_list": beta_list,
            "p_hat_list": p_hat_list,
            "mask_list": mask_list,
            "delta_list": delta_list,
            "confusion_matrix_list": confusion_matrix_list,
            "saved_r2_train": posterior_summary['r2_train'],
            "saved_r2_test": posterior_summary['r2_test'],
            "mean_parameter_r2": mean_parameter_r2
        }
        self.results.append(run_result)

        if true_beta_list is None:
            print("No true_beta_list available; skipping confusion matrix metrics.")
        else:
            for m, cm in enumerate(confusion_matrix_list):
                print(f"Confusion Matrix Modality {m + 1}:")
                print(cm)
                self.print_confusion_metrics(cm)

        return run_result

def display_selection(p_hat, x_dim, y_dim):
    plt.imshow(np.array(p_hat).reshape((x_dim, y_dim)), cmap="RdBu");
    plt.colorbar()

def display_modality1_across_experiments(all_experiments, x_dim, y_dim, nrows=3, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, experiment in enumerate(all_experiments[:nrows * ncols]):
        mask = experiment.results[0]["mask_list"][0]
        axes[i].imshow(np.array(mask).reshape(x_dim, y_dim), cmap="RdBu")
        axes[i].set_title(f"Experiment {i}")
        axes[i].axis("off")

    for j in range(len(all_experiments[:nrows * ncols]), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def display_modality2_across_experiments(all_experiments, x_dim, y_dim, nrows=3, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, experiment in enumerate(all_experiments[:nrows * ncols]):
        mask = experiment.results[0]["mask_list"][1]
        axes[i].imshow(np.array(mask).reshape(x_dim, y_dim), cmap="RdBu")
        axes[i].set_title(f"Experiment {i}")
        axes[i].axis("off")

    for j in range(len(all_experiments[:nrows * ncols]), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def display_r2_train_across_experiments(all_experiments, nrows=3, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, experiment in enumerate(all_experiments[:nrows * ncols]):
        r2_train = experiment.trainers[0].net.r2_train
        axes[i].plot(r2_train)
        axes[i].set_title(f"Experiment {i}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("R2 Train")
        axes[i].grid(True, alpha=0.3)

    for j in range(len(all_experiments[:nrows * ncols]), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def display_r2_eval_across_experiments(all_experiments, nrows=3, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, experiment in enumerate(all_experiments[:nrows * ncols]):
        r2_eval = experiment.trainers[0].net.r2_test
        axes[i].plot(r2_eval)
        axes[i].set_title(f"Experiment {i}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("R2 Eval")
        axes[i].grid(True, alpha=0.3)

    for j in range(len(all_experiments[:nrows * ncols]), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
