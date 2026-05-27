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
from transformers import ViTConfig, ViTModel

warnings.filterwarnings('ignore')

## Install R package BayesGPfit

# from GP_comp.GP import gp_eigen_value, gp_eigen_funcs_fast, generate_grids

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False, # Disable pinning to avoid the AcceleratorError
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
    )

    return train_loader, val_loader, n_train, n_val

class RegressionViT(nn.Module):
    def __init__(
        self, 
        num_classes=1, 
        dropout_prob=0.1, 
        image_size=(100, 200), 
        num_channels=3, 
        patch_size=8, 
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,  
        intermediate_size=2048
    ):  # Adjust image_size to 448, assume RGB images
        super(RegressionViT, self).__init__() #inherits class nn.Module

        self.config = ViTConfig( # creates a configuration object that defines ViT architecture and behavior
            dropout_prob=dropout_prob, # dropout rate used inside transformer
            image_size=image_size,  # Updated image size 
            num_channels=num_channels,  # Assuming RGB images
            patch_size=patch_size,  # Keeping the patch size
            hidden_size=hidden_size, # size of internal embedding vectors
            num_hidden_layers=num_hidden_layers, # number of transformer blocks
            num_attention_heads=num_attention_heads, # number of attention heads per block
            intermediate_size=intermediate_size, # size of teh feedforward layer inside of each transformer block)
        )

        self.vit = ViTModel(self.config) # creates ViT according to prespecified proportions

        self.preprocess = nn.Sequential( #small convolutional network that runs before image sent into vit
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1), #take input image and learn 32 feature maps from it
            nn.BatchNorm2d(32), #normalize those 32 channels
            nn.ReLU(), #add nonlinearity
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.config.num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.num_channels),
            nn.ReLU(),
        )

        # Adjust the size of the input layer of the regressor to match the output dimension of the ViT.
        # This requires calculating the number of output features from ViT, which depends on its configuration.
        # Assuming the dimensionality does not change for simplicity, but this may need to be adjusted based 
        # on the ViTModel implementation.
        
        # Define the new DNN architecture for the regressor
        self.fc1 = nn.Linear(self.config.hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
        self.fc5 = nn.Linear(4, num_classes)  # Output layer adjusted for num_classes
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(4)

    def forward(self, x):
        x = self.preprocess(x)
        outputs = self.vit(pixel_values=x)
        x = outputs.last_hidden_state[:, 0]  # Use the representation of the [CLS] token

        # Pass through the new DNN
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)  # No activation for the final layer in regression

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
        if self.device.type == "cuda":
            print(f"Using device: {self.device} ({torch.cuda.get_device_name(0)})", flush=True)
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}, version: {torch.backends.cudnn.version()}", flush=True)
        else:
            print("Using device: cpu", flush=True)
        torch.set_default_dtype(torch.float32)

        batch = next(iter(self.train_loader))
        x, _ = batch
        x = self._prepare_inputs(x)

        _, c, h, w = x.shape

        patch_size = 10   # choose a value that divides both h and w
        self.model = RegressionViT(
            num_classes=1,
            image_size=(h, w),
            num_channels=c,
            patch_size=patch_size,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            dropout_prob=0.1,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _prepare_inputs(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = x.to(self.device, non_blocking=self.device.type == "cuda").float()

        if x.ndim == 3:
            x = x.unsqueeze(1)   # [B, H, W] -> [B, 1, H, W]

        return x

    def _prepare_targets(self, y):
        return y.to(self.device, non_blocking=self.device.type == "cuda").float().view(-1, 1)

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

                x = self._prepare_inputs(x)
                y = self._prepare_targets(y)

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
                    x = self._prepare_inputs(x)
                    y = self._prepare_targets(y)

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
                    x = self._prepare_inputs(x)
                    y = self._prepare_targets(y)

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
            if epoch%10==0:
                print(f"Epoch {epoch}, r2_train={self.r2_train[-1]}", flush=True)
                if self.r2_val:
                    print(f"Epoch {epoch}, r2_val={self.r2_val[-1]}", flush=True)
                print(f"Epoch {epoch}, time={toc-tic:.2f}s", flush=True)

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
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def load_npz_data(self, sample_size):
        return self.load_simulation_data(sample_size=sample_size)


    def load_simulation_data(self, sample_size=1000, r2=0.8, in_feature_list=[100, 200], seed=42):
        n = sample_size
        dim1, dim2 = in_feature_list
    
        v_list1, v_list2, true_beta1, true_beta2, img1, img2, Y = simulate_data_circle_square(
            n=n,
            r2=r2,
            dim1=dim1,
            dim2=dim2,
            random_seed=seed
        )
    
        X1 = torch.from_numpy(img1).float().to("cpu")
        X2 = torch.from_numpy(img2).float().to("cpu")
    
        if X1.ndim == 2:
            X1 = X1.view(n, dim1, dim1)
    
        if X2.ndim == 2:
            X2 = X2.view(n, dim2, dim2)
    
        # Final canvas size
        H = max(dim1, dim2)
        W = dim1 + dim2
        X = torch.zeros((n, 1, H, W), dtype=X1.dtype, device=X1.device)
    
        # Place image 1 on the left
        X[:, 0, :dim1, :dim1] = X1
    
        # Place image 2 on the bottom-right
        X[:, 0, H - dim2:H, dim1:dim1 + dim2] = X2
    
        y = torch.from_numpy(Y).float().to("cpu")
        true_beta_list = [true_beta1, true_beta2]
    
        return X, y, true_beta_list

        
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

    def run_whole_experiment(self, sample_size, data_type='simulation', seed=42, true_r2=0.8,
                             val_ratio=0.5, in_feature_list=[100, 200], lr=1e-3, epochs=50):
        self.set_seed(seed)

        X_list, y, true_beta_list = self.load_simulation_data(
            sample_size=sample_size,
            r2=true_r2,
            in_feature_list=in_feature_list,
            seed=seed
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
