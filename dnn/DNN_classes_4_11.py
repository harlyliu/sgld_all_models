import os
import sys

# Add this model folder and the project root to the Python path.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for path in (THIS_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import argparse
import random
import time
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import copy
import torchvision
import warnings
import math
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings('ignore')

## Install R package BayesGPfit

from utils import generate_linear_data, plot_mse, plot_sigma_squared, plot_image
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, precision_score
import numpy as np
import torch.optim as optim
import time
from simulate_data import simulate_data_circle_square



import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

class mydata(Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i].view(1)



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

class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        nn_size =  [1024, 512, 256, 128]
        self.fc1 = nn.Linear(input_size, nn_size[0]) 
        self.fc2 = nn.Linear(nn_size[0], nn_size[1]) # takes input vector size 512, outputes vector of size 256
        self.fc3 = nn.Linear(nn_size[1], nn_size[2])
        self.fc4 = nn.Linear(nn_size[2], nn_size[3])
        self.fc5 = nn.Linear(nn_size[3], 1)
        self.dropout = nn.Dropout(0.5) # randomly drop half, necessary to prrevent overfitting
        # normalizes data from input layer. ensures more stable mean and variance
        self.bn1 = nn.BatchNorm1d(nn_size[0])  # Batch normalization after first layer
        self.bn2 = nn.BatchNorm1d(nn_size[1])  # Batch normalization after second layer
        self.bn3 = nn.BatchNorm1d(nn_size[2])  # Batch normalization after third layer
        self.bn4 = nn.BatchNorm1d(nn_size[3])   # Batch normalization after fourth layer

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class train_runner:
    def __init__(
        self,
        train_loader,
        val_loader,
        lr=1e-3,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr

        self.curr_epoch = 0
        self.total_epochs = 0

        self.start_save = 0
        self.save_every = 2
        self.N_saves = 100
        self.test_every = 10
        self.print_every = 10

        self.loss_train = []
        self.loss_val = []
        self.r2_train = []
        self.r2_val = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float32)
        first_batch = next(iter(train_loader))
        x, y = first_batch
        input_size = x.shape[1]
        self.model = DNNModel(input_size=input_size).to(self.device)

        self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _initialize_model(self):
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.train_loader))

            if len(batch) == 2:
                x, _ = batch
            else:
                (x, _), _ = batch

            if isinstance(x, (list, tuple)):
                x = x[0]

            x = x.to(self.device).float()

            if x.ndim == 3:
                x = x.unsqueeze(1)

            _ = self.model(x)

    def _compute_r2(self, y_true, y_pred):
        y_true = y_true.view(-1).float()
        y_pred = y_pred.view(-1).float()

        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

        return (1.0 - ss_res / ss_tot).item()

    def train(self, n_epochs):
        self.total_epochs += n_epochs
        tic = time.time()
        for epoch in range(self.curr_epoch, self.total_epochs):
            
            self.model.train()

            for batch in self.train_loader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    (x, _), y = batch

                if isinstance(x, (list, tuple)):
                    x = x[0]

                x = x.to(self.device).float()
                y = y.to(self.device).float().view(-1, 1)

                if x.ndim == 3:
                    x = x.unsqueeze(1)

                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.mse_loss(out, y, reduction="mean")
                loss.backward()
                self.optimizer.step()

            self.model.eval()

            with torch.no_grad():
                train_sse = 0.0
                train_n = 0
                train_preds = []
                train_targets = []

                for batch in self.train_loader:
                    x, y = batch
                    x = x.to(self.device).float()
                    y = y.to(self.device).float().view(-1, 1)

                    pred = self.model(x)
                    train_sse += torch.sum((y - pred) ** 2).item()
                    train_n += y.shape[0]
                    train_preds.append(pred)
                    train_targets.append(y)

                train_preds = torch.cat(train_preds, dim=0)
                train_targets = torch.cat(train_targets, dim=0)

                self.loss_train.append(train_sse / train_n)
                self.r2_train.append(self._compute_r2(train_targets, train_preds))

                val_sse = 0.0
                val_n = 0
                val_preds = []
                val_targets = []

                for batch in self.val_loader:
                    x, y = batch
                    x = x.to(self.device).float()
                    y = y.to(self.device).float().view(-1, 1)

                    pred = self.model(x)
                    val_sse += torch.sum((y - pred) ** 2).item()
                    val_n += y.shape[0]
                    val_preds.append(pred)
                    val_targets.append(y)

                val_preds = torch.cat(val_preds, dim=0)
                val_targets = torch.cat(val_targets, dim=0)

                self.loss_val.append(val_sse / val_n)
                self.r2_val.append(self._compute_r2(val_targets, val_preds))

            toc = time.time()
            if epoch % self.print_every == 0:
                print(f"Epoch {epoch}, r2_train={self.r2_train[-1]}")
                if self.r2_val:
                    print(f"Epoch {epoch}, r2_val={self.r2_val[-1]}")
                print(f"Epoch {epoch}, time={toc-tic:.2f}s")

        self.curr_epoch += n_epochs


        
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

    def load_npz_data(self, sample_size):
        return self.load_simulation_data(sample_size=sample_size)

    def load_simulation_data(self, sample_size=2000, r2=0.8, in_feature_list=[50, 100], seed=42):
        dim1, dim2 = in_feature_list
        v_list1, v_list2, true_beta1, true_beta2, img1, img2, y = simulate_data_circle_square(
            n=sample_size,
            r2=r2,
            dim1=dim1,
            dim2=dim2,
            random_seed=seed,
        )

        X1 = torch.from_numpy(img1.reshape(sample_size, -1)).float().to("cpu")
        X2 = torch.from_numpy(img2.reshape(sample_size, -1)).float().to("cpu")
        X = torch.cat([X1, X2], dim=1)
        y = torch.from_numpy(y).float().to("cpu")
        true_beta_list = [true_beta1, true_beta2]
        return X, y, true_beta_list


    def run_whole_experiment(self, seed=42, sample_size=2000, true_r2=0.8,
                             val_ratio=0.5, in_feature_list=[100, 200], lr=1e-3, epochs=50):
        self.set_seed(seed)
        X_list, y, true_beta_list = self.load_simulation_data(
            sample_size=sample_size,
            r2=true_r2,
            in_feature_list=in_feature_list,
            seed=seed,
        )

            
        train_loader, val_loader, n_train, n_val = create_dataloaders(
            X_list, y, val_ratio=val_ratio, batch_size=64
        )

        trainer = train_runner(
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
        )

        trainer.train(epochs)

        self.trainers.append(trainer)
        self.true_betas.append(true_beta_list)

        print(f"Sample Size: {sample_size}")
        print(f"Average r2_train: {np.mean(trainer.r2_train[-1:])}")
        print(f"Average r2_val: {np.mean(trainer.r2_val[-1:])}")
        return trainer

