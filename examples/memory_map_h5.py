import os
import sys
import argparse
import numpy as np

import tqdm

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from ivon import IVON as IBLR

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_test, flatten, predict_nll_hess, train_model, predict_train2
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_optim, get_pred_vars_laplace

import h5py
import json

def get_args():
    parser = argparse.ArgumentParser(description='Plotting Memory Maps')

    # Experiment
    parser.add_argument('--name_exp', default='memory_maps', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MOON', choices=['MNIST', 'FMNIST', 'CIFAR10', 'MOON'])
    parser.add_argument('--moon_noise', default = 0.05, type=float, help='desired noise for moon')
    parser.add_argument('--model', default='small_mlp',choices=['large_mlp', 'lenet', 'small_mlp', 'cnn_deepobs', 'nn'])

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

def get_optimizer(retrain=False):
    if retrain:
        lr = args.lr_retrain
    else:
        lr = args.lr
    if args.optimizer == 'adam':
        optim = Adam(net.parameters(), lr=lr, weight_decay=0)
    elif args.optimizer == 'iblr':
        optim = IBLR(net.parameters(), lr=lr, mc_samples=4, ess=n_train, weight_decay=1e-3,
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


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    dir = 'h5_files/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)

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
    for _ in tqdm.tqdm(list(range(args.epochs))):
        if args.optimizer == 'iblr':
            net, optim = train_one_epoch_iblr(net, optim, device)
        else:
            net, optim = train_one_epoch_sgd_adam(net, optim, device)

        test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
        test_nll_lst.append(test_nll)

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
    residuals, probs, lambdas, train_acc, train_nll = predict_train2(net, trainloader_eval, nc, tr_targets, device)
    print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

    # Evaluate on test data
    test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
    print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

    # Compute prediction variances
    vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, device, version='kfac')

    # Compute and store sensitivities
    sensitivities = np.asarray(residuals) * np.asarray(lambdas) * np.asarray(vars)
    sensitivities = np.sum(np.abs(sensitivities), axis=-1)

    if args.dataset == 'MOON':

        scores_dict = {'X_train': ds_train.tensors[0],
                    'y_train': ds_train.tensors[1],
                    'sensitivities': sensitivities,
                    'bpe': residuals_summary,
                    'bls': lev_scores_summary}
    else:
        scores_dict = {'sensitivities': sensitivities,
                    'bpe': residuals_summary,
                    'bls': lev_scores_summary}

    # Start of LOO experiment
    indices = np.arange(0, n_train, 1)
    indices_retrain = indices[0:args.n_retrain]

    # Retrain with one example removed
    softmax_deviations = np.zeros((args.n_retrain, nc))
    datapoint = []
    for i in tqdm.tqdm(list(range(args.n_retrain))):
        print('\nRemoved example ', i)

        # Warmstarting
        vector_to_parameters(w_star, net.parameters())
        net = net.to(device)

        # Remove one example from training set
        idx_removed = indices_retrain[i]
        idx_remain = np.setdiff1d(np.arange(0, n_train, 1), idx_removed)

        if args.dataset == 'MOON':
            # Extract the input tensor of the example to remove, convert to a list for comparison
            X_removed = ds_train[idx_removed][0].tolist()  # Flatten the tensor to a list for comparison

            # Use list comprehension to filter out the example
            ds_train_perturbed_list = [(x.tolist(), y) for x, y in ds_train if x.tolist() != X_removed]
            ds_removed_list = [(x.tolist(), y.numpy()) for x, y in ds_train if x.tolist() == X_removed]
            datapoint.append(ds_removed_list[0])
            
            ds_train_perturbed = [(torch.tensor(x).to(device), torch.tensor(y).to(device)) for x, y in ds_train_perturbed_list]

            X_removed = torch.tensor(X_removed)
        else:
            # For other datasets (like MNIST, CIFAR10), you can use the default removal
            ds_train_perturbed = Subset(ds_train, idx_remain)
        
        trainloader_retrain = get_quick_loader(DataLoader(ds_train_perturbed, batch_size=args.bs, shuffle=False), device=device)

        # Retraining
        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
        optim = get_optimizer()
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lrmin_retrain)

        net, losses = train_model(net, criterion, optim, scheduler, trainloader_retrain, args.epochs_retrain, n_train-1, _, device, args.optimizer == 'adam')

        net.eval()
        with torch.no_grad():
            if args.dataset == 'MOON':
                logits_wminus = net(X_removed.to(device))
            elif device == 'cuda':
                X_removed = transform_train(torch.asarray(ds_train.data[idx_removed]).numpy()).cuda()
                logits_wminus = net(X_removed.expand((1, -1, -1, -1)).to(device))
            else:
                X_removed = transform_train(torch.asarray(ds_train.data[idx_removed]).numpy())
                logits_wminus = net(X_removed.expand((1, -1, -1, -1)).to(device))
            probs_wminus = torch.softmax(logits_wminus, dim=-1).cpu().numpy()
            softmax_deviations[i] = probs_wminus - probs[idx_removed]

    # L1-norm
    softmax_deviations = np.sum(np.abs(softmax_deviations), axis=-1)

    # Save softmax deviations by removing an example and retraining (baseline)
    retrain_dict = {'indices_retrain': indices_retrain,
                    'softmax_deviations': softmax_deviations,
                     }

    with h5py.File(dir + args.name_exp + '_memory_maps.h5', 'w') as hf:
        scores_group = hf.create_group('scores')
        
        for key, value in scores_dict.items():
            scores_group.create_dataset(key, data=value)
        
        for key, value in retrain_dict.items():
            scores_group.create_dataset(key, data=value)
        
        # Save config as a JSON string
        config_group = hf.create_group('config')
        config_group.create_dataset('config_data', data=config_json)