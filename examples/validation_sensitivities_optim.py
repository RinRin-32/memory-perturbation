import os
import sys
import pickle
import argparse
import numpy as np

import tqdm

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, Adam, AdamW

from ivon import IVON as IBLR

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_train, predict_test, flatten, get_estimated_nll, train_network, train_iblr
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_laplace, get_pred_vars_optim


def get_args():
    parser = argparse.ArgumentParser(description='Compare sensitivities to true deviations in softmax outputs from retraining.')

    # Experiment
    parser.add_argument('--name_exp', default='mnist_mlp_ibr', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'FMNIST', 'CIFAR10', 'MOON'])
    parser.add_argument('--model', default='large_mlp',choices=['large_mlp', 'lenet', 'cnn_deepobs'])

    # Optimization
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'iblr','adam', 'adamw'])
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=1e-4, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--delta', default=60, type=float, help='L2-regularization parameter')

    # IBLR
    parser.add_argument('--hess_init', default=0.1, type=float, help='Hessian initialization')

    # Retraining
    parser.add_argument('--lr_retrain', default=1e-3, type=float, help='retraining: learning rate')
    parser.add_argument('--lrmin_retrain', default=1e-4, type=float, help='retraining: min learning rate scheduler')
    parser.add_argument('--epochs_retrain', default=300, type=int, help='retraining: number of epochs')
    parser.add_argument('--n_retrain', default=1000, type=int, help='number of retrained examples')

    # Variance computation
    # Variance computation
    parser.add_argument('--optim_var', dest='optim_var', action='store_true', help='variance from optimizer (IBLR, Adam)')
    parser.add_argument('--var_version', default='kfac', choices=['kfac', 'diag'], help='Laplace-GGN matrix structure', )
    parser.add_argument('--bs_jacs', default=50, type=int, help='Jacobian batch size for variance computation')
    parser.set_defaults(optim_var=False)
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
        optim = IBLR(net.parameters(), lr=args.lr, mc_samples=1, ess=n_train, weight_decay=args.delta/n_train,
                      beta1=0.9, beta2=0.99999, hess_init=args.hess_init)
    elif args.optimizer == 'sgd':
        optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError
    return optim

def get_prediction_vars(optim, device):
    if args.optim_var:  # Variances from optimizer state
        if args.optimizer == 'adam':
            sigma_sqrs = get_covariance_from_adam(optim, args.delta, n_train)
        elif args.optimizer == 'iblr':
            sigma_sqrs = get_covariance_from_iblr(optim)
        else:
            raise NotImplementedError
        sigma_sqrs = torch.asarray(flatten(sigma_sqrs)).to(device)
        vars = get_pred_vars_optim(net, trainloader_vars, sigma_sqrs, device)

    else:  # Laplace variance approximation
        if args.var_version == 'kfac':
            vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, device, version='kfac')
        elif args.var_version == 'diag':
            vars = get_pred_vars_laplace(net, trainloader_vars, args.delta, nc, device, version='diag')
        else:
            raise NotImplementedError

    return vars, optim


if __name__ == "__main__":
    args = get_args()
    print(args)

    evaluate_epochs = [1,2,5,10,20,30,40,50,75,100,150,200,250,300]

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    #device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    #if device == 'mps' and not args.optim_var:
    #    device = 'cpu'
    device = 'cuda'
    print('device', device)

    # Loss
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    # Data
    ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True)
    if args.dataset != 'MOON':
        input_size = len(ds_train.data[0, :])**2
        nc = max(ds_train.targets) + 1
        tr_targets, te_targets = torch.asarray(ds_train.targets), torch.asarray(ds_test.targets)
    else:
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lrmin)

    # Training
    test_accs, test_nlls, epochs_list = [], [], []
    for _ in tqdm.tqdm(list(range(args.epochs))):
        # Train for one epoch
        if args.optimizer == 'iblr':
            net, optim = train_one_epoch_iblr(net, optim, device)
        else:
            net, optim = train_one_epoch_sgd_adam(net, optim, device)

    # Evaluate on training data; residuals and logits
    residuals, prob, logits, train_acc, train_nll = predict_train(net, trainloader_eval, nc, tr_targets, device, return_logits=True)
    print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")
    print("Computing sensitivities...")

    # Prediction variances
    vars, optim = get_prediction_vars(optim, device)

    w_star = parameters_to_vector(net.parameters()).detach().cpu().clone()

    residuals_list, vars_list, logits_list, probs = np.asarray(residuals), np.asarray(vars), np.asarray(logits), np.asarray(prob)

    sensitivities = residuals_list * vars_list
    sensitivities = np.sum(np.abs(sensitivities), axis=-1)
    bpe = np.sum(np.abs(residuals_list), axis=-1)
    bls = np.sum(np.asarray(vars_list), axis=-1)

    def min_max_scale(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val)
    
    bpe = min_max_scale(bpe)
    bls = min_max_scale(bls)

    print(f'BPE: {bpe.max()} {bpe.min()}, BLS: {bls.max()} {bls.min()}')

    scores_dict = {'sensitivities': sensitivities,
                   'bpe': bpe,
                   'bls': bls}
    dir = 'pickles/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    with open(dir + args.name_exp + '_scores_optim.pkl', 'wb') as f:
        pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)

    # Random subsampling of examples for retraining
    indices = np.arange(0, n_train, 1)
    np.random.shuffle(indices)
    indices_retrain = indices[0:args.n_retrain]

    # Retrain with one example removed
    softmax_deviations = np.zeros((args.n_retrain, nc))
    for i in range(args.n_retrain):
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
            ds_train_perturbed = [(torch.tensor(x).to(device), torch.tensor(y).to(device)) for x, y in ds_train_perturbed_list]

            X_removed = torch.tensor(X_removed)
        else:
            # For other datasets (like MNIST, CIFAR10), you can use the default removal
            ds_train_perturbed = Subset(ds_train, idx_remain)

        trainloader_retrain = get_quick_loader(DataLoader(ds_train_perturbed, batch_size=args.bs, shuffle=True), device=device)

        # Retraining
        if args.optimizer == 'iblr':
            net, losses = train_iblr(net, criterion, optim, scheduler, trainloader_retrain, args.lr_retrain, args.lrmin_retrain, args.epochs_retrain, n_train-1, args.delta, device=device)
        else:
            net, losses = train_network(net, trainloader_retrain, args.lr_retrain, args.lrmin_retrain, args.epochs_retrain, n_train-1, args.delta, device=device)

        # Evaluate softmax deviations
        net.eval()
        with torch.no_grad():
            if args.dataset == 'MOON':
                X_removed = X_removed.unsqueeze(0)
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
    with open('pickles/' + args.name_exp + '_retrain_optim.pkl', 'wb') as f:
        pickle.dump(retrain_dict, f, pickle.HIGHEST_PROTOCOL)