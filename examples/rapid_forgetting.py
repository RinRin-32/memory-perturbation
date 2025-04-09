## randomly splice dirty moon dataset, this experiment will show cases of using Sensitivity to find unpredictable/uncertain cases!

import os
import sys
import pickle
import argparse
import numpy as np
import json
import h5py
import tqdm

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from ivon import IVON as IBLR
import matplotlib.pyplot as plt

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_test, flatten, predict_nll_hess, train_model, predict_train2, get_estimated_nll
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_optim, get_pred_vars_laplace

import random

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

    #print(f"Contour plot saved as {save_path}")

    return xx, yy, Z

def get_args():
    parser = argparse.ArgumentParser(description='Plotting Memory Maps')

    # Experiment
    parser.add_argument('--name_exp', default='mnist_lenet_ibr', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'FMNIST', 'CIFAR10', 'MOON'])
    parser.add_argument('--moon_noise', default = 0.2, type=float, help='desired noise for moon')
    parser.add_argument('--model', default='lenet',choices=['large_mlp', 'lenet', 'small_mlp', 'cnn_deepobs', 'nn'])

    # Optimization
    parser.add_argument('--optimizer', default='iblr', choices=['iblr']) #no adam support yet
    parser.add_argument('--lr', default=2, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=1e-3, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--delta', default=60, type=float, help='L2-regularization parameter')

    # IBLR
    parser.add_argument('--hess_init', default=0.1, type=float, help='Hessian initialization')

    # Retraining
    parser.add_argument('--lr_retrain', default=2, type=float, help='retraining: learning rate')
    parser.add_argument('--lrmin_retrain', default=1e-3, type=float, help='retraining: min learning rate scheduler')
    parser.add_argument('--epochs_retrain', default=1000, type=int, help='retraining: number of epochs')
    parser.add_argument('--n_retrain', default=1000, type=int, help='number of retrained examples')

    # Variance computation
    parser.add_argument('--bs_jacs', default=50, type=int, help='Jacobian batch size for variance computation')
    return parser.parse_args()

def train_one_epoch_iblr(net, optim, device, trainloader):
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

def train_one_epoch_sgd_adam(net, optim, device, trainloader):
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

def get_optimizer(retrain=False):
    if retrain:
        lr = args.lr_retrain
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optim = Adam(net.parameters(), lr=lr, weight_decay=0)
    elif args.optimizer == 'iblr':
        optim = IBLR(net.parameters(), lr=lr, mc_samples=1, ess=n_train, weight_decay=1e-3,
                      beta1=0.9, beta2=0.99999, hess_init=args.hess_init)
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

def make_dirty_dataset_from_sensitivities(ds_train, sensitivities, noise_rate=0.2):
    ds_dirty = list(ds_train)  # copy
    num_samples = len(ds_dirty)
    num_noisy = int(noise_rate * num_samples)

    # Get indices of top-N most sensitive points
    sorted_indices = np.argsort(-sensitivities)  # descending
    noisy_indices = sorted_indices[:num_noisy]

    all_classes = list(set(target for _, target in ds_dirty))
    for idx in noisy_indices:
        _, original_label = ds_dirty[idx]
        new_label = random.choice([c for c in all_classes if c != original_label])
        ds_dirty[idx] = (ds_dirty[idx][0], new_label)
    
    return ds_dirty

if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.dataset == 'MOON' and (args.model == 'lenet' or args.model == 'cnn_deepobs'):
        raise NotImplementedError(f'{args.model} does not support the moon dataset')

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    # Loss
    criterion = nn.CrossEntropyLoss().to(device)

    output_dir = "h5_files/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.name_exp}_test_epoch.h5")

    # Data
    if args.dataset != 'MOON':
        ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True)
        input_size = len(ds_train.data[0, :])**2
        nc = max(ds_train.targets) + 1
        tr_targets, te_targets = torch.asarray(ds_train.targets), torch.asarray(ds_test.targets)
    else:
        ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True, noise=args.moon_noise)
        input_size = ds_train[0][0].numel()
        nc = len(torch.unique(torch.asarray([target for _, target in ds_train])))
        tr_targets = torch.asarray([target for _, target in ds_train])
        te_targets = torch.asarray([target for _, target in ds_test])
    n_train = len(ds_train)

    # Model
    net = get_model(args.model, nc, input_size, device, seed)

    # Dataloaders
    trainloader = get_quick_loader(DataLoader(ds_train, batch_size=args.bs), device=device) # training
    trainloader_eval = DataLoader(ds_train, batch_size=args.bs, shuffle=False) # train evaluation
    testloader_eval = DataLoader(ds_test, batch_size=args.bs, shuffle=False) # test evaluation
    trainloader_vars = DataLoader(ds_train, batch_size=args.bs_jacs, shuffle=False) # variance computation

    clean_scores = {}
    dirty_scores = {}
    # Optimizer
    optim = get_optimizer()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    config = {
        "input_size": input_size,
        "nc": int(nc.item()) if isinstance(nc, torch.Tensor) else nc,
        "model": args.model,
        "device": device,
        "optimizer": args.optimizer,
        "optimizer_params": {
            key: value
            for key, value in vars(args).items()
            if (key.startswith('lr') or key.startswith('delta') or key.startswith('hess_init'))
        },
        "max_epochs": args.epochs,
        "loss_criterion": "CrossEntropyLoss",
        "n_retrain": args.n_retrain
    }
    config_json = json.dumps(config)

    residual_upper, leverage_upper = 0.,0.
    test_nll_lst, loocv_lst = [], []

    for epoch in tqdm.tqdm(list(range(args.epochs*2))):
        if args.optimizer == 'iblr':
            net, optim = train_one_epoch_iblr(net, optim, device, trainloader)
        else:
            net, optim = train_one_epoch_sgd_adam(net, optim, device, trainloader)

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
        residual, probs, lambdas, logits, train_acc, train_nll = predict_train2(net, trainloader_eval, nc, tr_targets, device, return_logits=True)
        print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

        # Evaluate on test data
        test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
        print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

        vars, optim = get_prediction_vars(optim, device)

        #sensitivities = np.asarray(residuals) * np.asarray(lambdas) * np.asarray(vars)
        #sensitivities = np.sum(np.abs(sensitivities), axis=-1)
        vars_diag = vars.diagonal(dim1=1, dim2=2)
        sensitivities = np.asarray(residuals) * np.asarray(lambdas) * np.asarray(vars_diag)
        sensitivities = np.sum(np.abs(sensitivities), axis=-1)


        if args.dataset == 'MOON':
            xx, yy, Z = plot_contour(net, ds_train)
            decision_boundary = {"xx": xx, "yy": yy, "Z": Z}
            scores_dict = {
                        'sensitivities': sensitivities,
                        'decision_boundary': decision_boundary,
                        'bpe': residuals_summary,
                        'bls': lev_scores_summary}
        else:
            scores_dict = {'sensitivities': sensitivities,
                           'bpe': residuals_summary,
                           'bls': lev_scores_summary}

        clean_scores[epoch] = scores_dict

        
        # can do this if you want to show the model "forgetting"
        if epoch == args.epochs:
            initial_ds_train = ds_train
            ds_dirty = make_dirty_dataset_from_sensitivities(ds_train, noise_rate=0.2, sensitivities=sensitivities)
            dirty_tr_targets = torch.asarray([target for _, target in ds_dirty])
            dirty_trainloader = get_quick_loader(DataLoader(ds_dirty, batch_size=args.bs), device=device) # dirty train loader
            dirty_trainloder_eval = DataLoader(ds_dirty, batch_size=args.bs, shuffle=False)
            trainloader = dirty_trainloader
            tr_targets = dirty_tr_targets
            trainloader_eval = dirty_trainloder_eval
            ds_train = ds_dirty

    X_dirty = torch.stack([x for x, _ in ds_dirty])  # Stack inputs into a single tensor
    y_dirty = torch.tensor([y for _, y in ds_dirty])  # Convert labels to a tensor

    with h5py.File(output_file, 'w') as f:
        if args.dataset == 'MOON':
            coord_group = f.create_group('coord')
            x_coord = coord_group.create_dataset('X_train', data=initial_ds_train.tensors[0])
            y_coord = coord_group.create_dataset('y_train', data=initial_ds_train.tensors[1])
            x_coord_dirty = coord_group.create_dataset('X_train_dirty', data=X_dirty)
            y_coord_dirty = coord_group.create_dataset('y_train_dirty', data=y_dirty)
        config_group = f.create_group("config")
        config_group.create_dataset('config_data', data=config_json)

        scores_group = f.create_group('clean_scores')

        for epoch, data in clean_scores.items():
            epoch_group_name = f"epoch_{epoch}"
            epoch_group = scores_group.create_group(epoch_group_name)

            for key, value in data.items():
                if isinstance(value, dict):
                    sub_group = epoch_group.create_group(key) if key not in epoch_group else epoch_group[key]
                    for sub_key, sub_value in value.items():
                        sub_group.create_dataset(sub_key, data=sub_value)
                else:
                    epoch_group.create_dataset(key, data=value)
            
    print(f"Saved rapid forgetting test at {output_file}")