import os
import sys
import argparse
import numpy as np
import json

import tqdm
import math

import h5py

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
import torch.nn.functional as F

import torch.optim as optim

from ivon import IVON as IBLR

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_test, flatten, predict_nll_hess, predict_train2, get_estimated_nll
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_optim, get_pred_vars_laplace

import matplotlib.pyplot as plt

def plot_contour(model, dataset, save_path="contour_plot.png", resolution=0.01, batch_size=10000):
    model.eval()  # Set to evaluation mode

    # Extract features and labels from dataset
    X, y = zip(*dataset)
    X = torch.stack(X).numpy()
    y = torch.stack(y).numpy()

    # Define the plot area
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid over the feature space
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # Flatten grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Move model to CPU to save CUDA memory
    device = next(model.parameters()).device
    model_cpu = model.to("cpu")

    # Predict in batches to save memory
    predictions = []
    with torch.no_grad():
        for i in range(0, len(grid_points), batch_size):
            batch = torch.tensor(grid_points[i:i + batch_size], dtype=torch.float32)
            batch_preds, _ = model_cpu(batch)
            batch_preds = torch.squeeze((batch_preds) > 0.5).float()
            predictions.append(batch_preds)

    # Combine results
    Z = np.concatenate(predictions).reshape(xx.shape)

    # Move model back to its original device
    model.to(device)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)  # Smooth filled contour
    plt.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.7)  # Clear boundary lines

    # Scatter plot of actual data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='white', alpha=0.5)

    # Add legend and labels
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.title('Decision Boundary of Multi-Class Classifier')

    # Save the plot instead of showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"Contour plot saved as {save_path}")

    return xx, yy, Z

def get_args():
    parser = argparse.ArgumentParser(description='Plotting Sensitivity over Epoch')

    # Experiment
    parser.add_argument('--name_exp', default='visualizer', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MOON', choices=['MOON'])
    parser.add_argument('--model', default='nn_single',choices=['nn_single'])

    # Optimization
    parser.add_argument('--optimizer', default='iblr', choices=['iblr', 'adam'])
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--delta', default=60, type=float, help='L2-regularization parameter')

    # IBLR
    parser.add_argument('--hess_init', default=0.1, type=float, help='Hessian initialization')

    # Variance computation
    parser.add_argument('--bs_jacs', default=50, type=int, help='Jacobian batch size for variance computation')

    return parser.parse_args()

def train_one_epoch_iblr(net, optim, device):
    net.train()
    running_loss = 0
    for X, y in trainloader:
        X, y = X.to(device), y.to(device)
        with optim.sampled_params(train=True):
            optim.zero_grad()
            fs = net(X)
            loss = criterion(fs, y)
            loss.backward()
        optim.step()
        running_loss += loss.item()
    scheduler.step()
    return net, optim

def train_one_epoch_sgd_adam(net, optim, device):
    net.train()
    running_loss = 0
    for X, y in trainloader:
        X, y = X.to(device), y.to(device)
        def closure():
            optim.zero_grad()
            fs = net(X)
            loss_ = criterion(fs, y)
            if args.optimizer == 'adamw':
                reg_ = 0
            else:
                p_ = parameters_to_vector(net.parameters())
                reg_ = 1/2 * args.delta * p_.square().sum()
            loss = loss_ + (1/n_train)*reg_
            loss.backward()
            return loss, fs
        loss, fs = optim.step(closure)
        running_loss += loss.item()
    scheduler.step()
    return net, optim

def get_optimizer():
    if args.optimizer == 'adam':
        optim = Adam(net.parameters(), lr=args.lr, weight_decay=0)
    elif args.optimizer == 'adamw':
        optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.delta / n_train)
    elif args.optimizer == 'iblr':
        optim = IBLR(net.parameters(), lr=args.lr, mc_samples=1, ess=n_train, weight_decay=1e-3,
                      beta1=0.9, beta2=0.99999, hess_init=args.hess_init)
    elif args.optimizer == 'sgd':
        optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError
    return optim

def get_prediction_vars(optim, device):
    if args.optimizer == 'adam':
        sigma_sqrs = get_covariance_from_adam(optim, args.delta, n_train)
    elif args.optimizer == 'iblr':
        sigma_sqrs = get_covariance_from_iblr(optim)
    else:
        raise NotImplementedError
    sigma_sqrs = torch.asarray(flatten(sigma_sqrs)).to(device)
    vars = get_pred_vars_optim(net, trainloader_vars, sigma_sqrs, device, tensor=True)

    return vars, optim

def induced_label_noise(optimizer, model, vis_loader, n_train, nc, bs):
    model.eval()
    mc_samples = n_train
    label_noise_all = np.zeros((n_train, nc))
    for i, (x, y) in enumerate(vis_loader):
        x, y = x.to(device), y.to(device)
        res = torch.zeros(len(y), nc)
        temp, _ = model(x)
        at_mean = temp.cpu().detach().numpy()
        for _ in range(mc_samples):
            with optimizer.sampled_params(train = False):
                temp, _ = model(x)
                at_sample = temp.cpu().detach().numpy()
                #print(at_sample)
                res = res + at_sample - at_mean
        res = res/mc_samples
        label_noise_all[bs*i: bs*i + len(y), :]= res
    return label_noise_all

def extract_sigmoid(optimizer, model, vis_loader, n_train, bs):
    logits_all = np.zeros(n_train)
    sig_input = np.zeros(n_train)
    for i, (x,y) in enumerate(vis_loader):
        x, y = x.to(device), y.to(device)
        with optimizer.sampled_params(train = False):
            logits, sig_in = model(x)
        logits_all[bs*i: bs*i + len(y)] = logits.cpu().detach().squeeze(1).numpy()
        sig_input[bs*i: bs*i + len(y)] = sig_in.cpu().detach().squeeze(1).numpy()
    return logits_all, sig_input


def dsoftmax(z):
    s = F.softmax(z)
    extends=s.unsqueeze(2)
    #print(s.shape,torch.diag_embed(s).shape, extends.transpose(1,2).shape, (extends @ extends.transpose(1,2)).shape)
    return torch.diag_embed(s) - extends @ extends.transpose(1,2)

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Device
    #device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    print('device', device)

    # Loss
    
    criterion = nn.BCELoss().to(device)

    output_dir = "h5_files/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.name_exp}_step_single.h5")

    # Data
    ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True, noise=0.05, n_samples=1000)
    input_size = ds_train[0][0].numel()
    nc = len(torch.unique(torch.asarray([target for _, target in ds_train])))
    tr_targets = torch.asarray([target for _, target in ds_train])
    te_targets = torch.asarray([target for _, target in ds_test])
    n_train = len(ds_train)
    n_samples = len(ds_train)

    # Model
    net = get_model(args.model, nc, input_size, device, seed)

    # Dataloaders
    trainloader = get_quick_loader(DataLoader(ds_train, batch_size=args.bs), device=device) # training
    trainloader_eval = DataLoader(ds_train, batch_size=args.bs, shuffle=False) # train evaluation
    testloader_eval = DataLoader(ds_test, batch_size=args.bs, shuffle=False) # test evaluation
    trainloader_vars = DataLoader(ds_train, batch_size=args.bs_jacs, shuffle=False) # variance computation

    vis_loader = DataLoader(dataset=ds_train, batch_size=args.bs, shuffle=False)
    # Optimizer
    optim = get_optimizer()

    mc_samples = 1

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    config = {
        "input_size": input_size,
        "nc": int(nc.item()) if isinstance(nc, torch.Tensor) else nc,
        "model": args.model,
        "dataset": args.dataset,
        "device": device,
        "optimizer": args.optimizer,
        "optimizer_params": {
            key: value
            for key, value in vars(args).items()
            if (key.startswith('lr') or key.startswith('delta') or key.startswith('hess_init'))
        },
        "max_epochs": args.epochs,
        "total_batch": math.ceil(n_train/args.bs),
        "loss_criterion": "CrossEntropyLoss",
        "batch_size": args.bs
    }

    residual_upper, leverage_upper = 0.,0.

    penultimate_features_list = []
    labels_list = []

    residual_upper, leverage_upper = 0.,0.

    all_scores = {}
    all_result = {}

    curr = 0

    for epoch in tqdm.tqdm(list(range(args.epochs+1))):
        running_loss = 0
        for _ in range(mc_samples):
            for X, y in trainloader:
                num_classes = nc
                all_noise = induced_label_noise(optim, net, vis_loader, n_train, 2, args.bs)

                logits_all, sig_input = extract_sigmoid(optim, net, vis_loader, n_train, args.bs)

                induced_noise = all_noise

                all_noise = [np.linalg.norm(x,2) for x in all_noise]

                if args.dataset == 'MOON':
                    xx, yy, Z = plot_contour(net, ds_train)
                    decision_boundary = {"xx": xx, "yy": yy, "Z": Z}

                if args.dataset == 'MOON':
                    scores_dict = {
                        'noise': all_noise,
                        'all_noise': induced_noise,
                        'decision_boundary': decision_boundary,
                        'logits': logits_all,
                        'sig_input': sig_input
                    }
                else:
                    scores_dict = {
                        'noise': all_noise,
                        'all_noise': induced_noise,
                    }
                result_dict = {
                    'step': curr,
                }

                all_scores[curr] = scores_dict
                all_result[curr] = result_dict
                net.train()
                #X, y = X.to(device), y.to(device)
                X = torch.tensor(X, dtype=torch.float32, device=device)
                y = torch.tensor(y, dtype=torch.float32, device=device)
                with optim.sampled_params(train=True):
                    optim.zero_grad()
                    fs, _ = net(X)
                    loss = criterion(torch.squeeze(fs), y)
                    loss.backward()
                optim.step()
                running_loss += loss.item()
                scheduler.step()
                curr += 1

    config_json = json.dumps(config)

    with h5py.File(output_file, 'w') as f:
        if args.dataset == 'MOON':
            coord_group = f.create_group('coord')
            x_coord = coord_group.create_dataset('X_train', data=ds_train.tensors[0])
            y_coord = coord_group.create_dataset('y_train', data=ds_train.tensors[1])

        config_group = f.create_group("config")
        config_group.create_dataset('config_data', data=config_json)

        scores_group = f.create_group('scores')

        for epoch, data in all_scores.items():
            epoch_group_name = f"step_{epoch}"
            epoch_group = scores_group.create_group(epoch_group_name)

            for key, value in data.items():
                if isinstance(value, dict):
                    sub_group = epoch_group.create_group(key) if key not in epoch_group else epoch_group[key]
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    epoch_group.create_dataset(key, data=value)

        result_group = f.create_group('results')

        for epoch, data in all_result.items():
            epoch_group_name = f"step_{epoch}"
            epoch_group = result_group.create_group(epoch_group_name)

            for key, value in data.items():
                if isinstance(value, dict):
                    sub_group = epoch_group.create_group(key) if key not in epoch_group else epoch_group[key]
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    epoch_group.create_dataset(key, data=value)
            
    print(f"Saved images, labels, and noise values to {output_file}")