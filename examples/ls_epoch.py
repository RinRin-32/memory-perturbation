import os
import sys
import argparse
import numpy as np
import json

import tqdm

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
            batch_preds = model_cpu(batch).argmax(dim=1).numpy()
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

    print(f"Contour plot saved as {save_path}")

    return xx, yy, Z

def get_args():
    parser = argparse.ArgumentParser(description='Plotting Sensitivity over Epoch')

    # Experiment
    parser.add_argument('--name_exp', default='visualizer', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'FMNIST', 'CIFAR10', 'MOON'])
    parser.add_argument('--model', default='small_mlp',choices=['large_mlp', 'lenet', 'small_mlp', 'cnn_deepobs', 'nn', 'linear_model', 'resnet34'])

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

def compute_labelnoise(train_loader, model, optimizer, device, num_classes, train_num, batch_size, mc):
    model.eval()
    label_noise_all = np.zeros((train_num, num_classes))
    labels_all = np.zeros(train_num)
    noises = np.zeros(train_num)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        label_noises = np.zeros((len(labels), num_classes))
        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            for _ in range(mc):
                with optimizer.sampled_params(train=False):
                    sample_logits = model(images)
                    sample_probs = F.softmax(sample_logits, dim=1)

                label_noise = sample_probs.cpu().numpy() - probs.cpu().numpy()
                label_noises = label_noises + label_noise/mc
        label_noise_all[batch_size*i: batch_size*i + len(labels), :]= label_noises
        labels_all[batch_size*i: batch_size*i + len(labels)] = labels.cpu().numpy()
        #noises[batch_size*i: batch_size*i + len(labels)] = noise_bool
    return label_noise_all, labels_all #, noises

def dsoftmax(z):
    s = F.softmax(z)
    extends=s.unsqueeze(2)
    #print(s.shape,torch.diag_embed(s).shape, extends.transpose(1,2).shape, (extends @ extends.transpose(1,2)).shape)
    return torch.diag_embed(s) - extends @ extends.transpose(1,2)
    
def compute_variance(model, optimizer, vis_loader, train_num, device, nc):
    num_classes = nc
    n_params = sum(p.numel() for p in model.parameters())
    # Forward pass for the entire batch
    variance_list = []
    d_softmax_list = []
    sqrt_cov_list = []
    
    for batch_idx, (X, y) in enumerate(vis_loader):
        X, y = X.to(device), y.to(device)
        output = model(X)  # Assuming output shape: (batch_size, 2)
        dsig = dsoftmax(output)
        # Initialize Jacobian for the entire batch: shape (batch_size, 2, n_params)
        batch_size = X.size(0)
        n_params = sum(p.numel() for p in model.parameters())
        jacobian = torch.zeros(batch_size, num_classes, n_params).to(device)
        # Compute Jacobian for each sample in the batch
        for j in range(batch_size):
            for i in range(num_classes):
                optimizer.zero_grad()
                output[j, i].backward(retain_graph=True)
                jacobian[j, i] = torch.cat([p.grad.flatten() for p in model.parameters()])

        # Extract required parameters from the optimizer
        delta = optimizer.param_groups[0]['weight_decay']
        hess = optimizer.param_groups[0]['hess']
        ess = optimizer.param_groups[0]['ess']
        lam = 1 / ess * (hess + delta)
        lam = torch.sqrt(lam)
        lam.to(device)
        sqrt_var = jacobian * lam
        variance_list.append(torch.norm(sqrt_var, dim = [1,2]).detach().cpu().numpy())
        sqrt_pred_var = torch.einsum('nck, nkp->ncp', dsig,sqrt_var)

        sqrt_cov_list.append(torch.norm(sqrt_pred_var, dim = [1,2]).detach().cpu().numpy())

        d_softmax_list.append(torch.norm(dsig, dim = [1,2]).detach().cpu().numpy())

    return variance_list, d_softmax_list, sqrt_cov_list

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
    
    criterion = nn.CrossEntropyLoss().to(device)

    output_dir = "h5_files/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.name_exp}_ls_epoch.h5")

    # Data
    ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True, noise=0.05)
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
        "loss_criterion": "CrossEntropyLoss",
    }

    residual_upper, leverage_upper = 0.,0.

    penultimate_features_list = []
    labels_list = []

    residual_upper, leverage_upper = 0.,0.

    all_scores = {}
    all_result = {}

    for epoch in tqdm.tqdm(list(range(args.epochs+1))):
        num_classes = nc
        all_noise, _ = compute_labelnoise(vis_loader, net, optim, device, num_classes, n_samples, args.bs, mc_samples)

        induced_noise = all_noise

        all_noise = [np.linalg.norm(x,2) for x in all_noise]

        index=list(range(n_samples))
        labels = tr_targets

        if args.dataset == 'MOON':
            xx, yy, Z = plot_contour(net, ds_train)
            decision_boundary = {"xx": xx, "yy": yy, "Z": Z}

        test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
    
        residuals, probs, logits, nll_hess, train_acc, train_nll = predict_nll_hess(net, trainloader_eval, nc, tr_targets, device)

        vars, optim = get_prediction_vars(optim, device)

        # Evaluate memory map criteria
        residuals_summary = torch.sqrt(torch.sum(residuals**2, dim=1)).detach().numpy() # l2norm
        lev_scores_full = torch.einsum('nij,nji->ni', vars, nll_hess)
        lev_scores_full = torch.clamp(lev_scores_full, 0.)
        lev_scores_summary = torch.sqrt(torch.sum(lev_scores_full**2, dim=1)).cpu().detach().numpy()

        leverage_upper = lev_scores_summary.max() if lev_scores_summary.max() > leverage_upper else leverage_upper
        residual_upper = residuals_summary.max() if residuals_summary.max() > residual_upper else residual_upper
        
        w_star = parameters_to_vector(net.parameters()).detach().cpu().clone()

        # Evaluate on training data; residuals and lambdas
        residuals, probs, lambdas, logits, train_acc, train_nll = predict_train2(net, trainloader_eval, nc, tr_targets, device, return_logits=True)
        print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

        # Evaluate on test data
        test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
        print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

        # Compute prediction variances
        vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, device, version='kfac')

        # Compute and store sensitivities
        sensitivities = np.asarray(residuals) * np.asarray(lambdas) * np.asarray(vars)
        sensitivities = np.sum(np.abs(sensitivities), axis=-1)

        estimated_nll = get_estimated_nll(nc, np.array([residuals]), np.array([vars]), logits, tr_targets)
        print(estimated_nll)

        if args.dataset == 'MOON':
            scores_dict = {
                'sensitivities': sensitivities,
                'bpe': residuals_summary,
                'bls': lev_scores_summary,
                'noise': all_noise,
                'all_noise': induced_noise,
                'decision_boundary': decision_boundary
            }
        else:
            scores_dict = {
                'sensitivities': sensitivities,
                'bpe': residuals_summary,
                'bls': lev_scores_summary,
                'noise': all_noise,
                'all_noise': induced_noise,
            }
        result_dict = {
            'epoch': epoch,
            'test_acc': test_acc,
            'test_nll': test_nll,
            'estimated_nll': estimated_nll
        }

        all_scores[epoch] = scores_dict
        all_result[epoch] = result_dict

        if args.optimizer == 'iblr':
            net, optim = train_one_epoch_iblr(net, optim, device)
        else:
            net, optim = train_one_epoch_sgd_adam(net, optim, device)

    config_json = json.dumps(config)

    with h5py.File(output_file, 'w') as f:
        if args.dataset == 'MOON':
            coord_group = f.create_group('coord')
            x_coord = coord_group.create_dataset('X_train', data=ds_train.tensors[0])
            y_coord = coord_group.create_dataset('y_train', data=ds_train.tensors[1])
        else:
            f.create_dataset("images", data=torch.stack([ds_train[i][0] for i in index]).numpy())  # Save sorted images
            f.create_dataset("labels", data=np.array(labels))  # Sorted labels
        config_group = f.create_group("config")
        config_group.create_dataset('config_data', data=config_json)

        scores_group = f.create_group('scores')

        for epoch, data in all_scores.items():
            epoch_group_name = f"epoch_{epoch}"
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
            epoch_group_name = f"epoch_{epoch}"
            epoch_group = result_group.create_group(epoch_group_name)

            for key, value in data.items():
                if isinstance(value, dict):
                    sub_group = epoch_group.create_group(key) if key not in epoch_group else epoch_group[key]
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    epoch_group.create_dataset(key, data=value)
            
    print(f"Saved MNIST images, labels, and noise values to {output_file}")