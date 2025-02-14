import os
import sys
import argparse
import numpy as np
import json
import math

import tqdm

import h5py

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, Adam, AdamW
import torch.nn.functional as F

import torch.optim as optim

from ivon import IVON as IBLR

sys.path.append("..")
from lib.models import get_model
from lib.datasets import get_dataset
from lib.utils import get_quick_loader, predict_test, flatten, predict_nll_hess, predict_train2
from lib.variances import get_covariance_from_iblr, get_covariance_from_adam, get_pred_vars_optim, get_pred_vars_laplace
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def get_args():
    parser = argparse.ArgumentParser(description='Plotting Sensitivity over Epoch')

    # Experiment
    parser.add_argument('--name_exp', default='visualizer', type=str, help='name of experiment')

    # Data, Model
    parser.add_argument('--dataset', default='MOON', choices=['MNIST', 'FMNIST', 'CIFAR10', 'MOON', 'MNIST_REDUX'])
    parser.add_argument('--moon_noise', default = 0.2, type=float, help='desired noise for moon')
    parser.add_argument('--model', default='small_mlp',choices=['large_mlp', 'lenet', 'small_mlp', 'cnn_deepobs', 'nn', 'linear_model', 'resnet34'])

    # Optimization
    parser.add_argument('--optimizer', default='iblr', choices=['iblr', 'adam'])
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lrmin', default=0.0, type=float, help='min learning rate of scheduler')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--delta', default=60, type=float, help='L2-regularization parameter')

    parser.add_argument('--ls', default=False, type=bool, help='traditional label smoothing')

    # IBLR
    parser.add_argument('--hess_init', default=0.1, type=float, help='Hessian initialization')

    # Retraining
    parser.add_argument('--lrmin_retrain', default=0.0, type=float, help='retraining: min learning rate scheduler')
    parser.add_argument('--n_retrain', default=800, type=int, help='number of retrained examples')

    # Variance computation
    parser.add_argument('--bs_jacs', default=50, type=int, help='Jacobian batch size for variance computation')

    # logging
    parser.add_argument('--log_step', default = 1, type=int, help='1 equates to logging on every step')

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
        optim = IBLR(net.parameters(), lr=args.lr, mc_samples=4, ess=n_train, weight_decay=args.delta/n_train,
                      beta1=0.9, beta2=0.99999, hess_init=args.hess_init)
    elif args.optimizer == 'sgd':
        optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError
    return optim


def plot_decision_boundary(model, X, device='cuda'):
    # Define the grid range
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Create the grid points
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict on the grid
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
        preds = model(grid_tensor).argmax(dim=1).cpu().numpy()
    
    # Reshape predictions to match the grid
    Z = preds.reshape(xx.shape)

    return xx, yy, Z

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

def save_visualization(dataset, num_samples=1000, filename="dataset_plot.png"):
    """Saves a 2D PyTorch dataset visualization to a file."""
    
    # Extract a subset of data
    X, y = zip(*[dataset[i] for i in range(min(len(dataset), num_samples))])  
    X = torch.stack(X)  # Ensure it's a tensor
    y = torch.tensor(y)  # Ensure labels are tensor
    
    # Create plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.numpy(), cmap='tab10', alpha=0.5)
    plt.colorbar(scatter, label="Class Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Visualization of Dataset")
    
    # Save image
    save_path = os.path.join("./" + filename)
    plt.savefig(save_path)
    plt.close()
    
    return save_path

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
    
def compute_variance(model, optimizer, vis_loader, train_num, device):
    num_classes = 10
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
    if args.ls:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    output_dir = "h5_files/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.name_exp}_ls_exp.h5")

    # Data
    ds_train, ds_test, transform_train = get_dataset(args.dataset, return_transform=True, noise=args.moon_noise)
    input_size = ds_train[0][0].numel()
    nc = len(torch.unique(torch.asarray([target for _, target in ds_train])))
    tr_targets = torch.asarray([target for _, target in ds_train])
    te_targets = torch.asarray([target for _, target in ds_test])
    n_train = len(ds_train)
    n_samples = len(ds_train)
    #usage: For quick look at how your data look on 2D plane
    #save_visualization(ds_train)

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

    mc_samples = 4

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    residual_upper, leverage_upper = 0.,0.

    penultimate_features_list = []
    labels_list = []

    for epoch in tqdm.tqdm(list(range(args.epochs))):
        if args.optimizer == 'iblr':
            net, optim = train_one_epoch_iblr(net, optim, device)
        else:
            net, optim = train_one_epoch_sgd_adam(net, optim, device)

    # Evaluate on training data; residuals and lambdas
    residuals, probs, lambdas, train_acc, train_nll = predict_train2(net, trainloader_eval, nc, tr_targets, device)
    print(f"Train Acc: {(100 * train_acc):>0.2f}%, Train NLL: {train_nll:>6f}")

    # Evaluate on test data
    test_acc, test_nll = predict_test(net, testloader_eval, nc, te_targets, device)
    print(f"Test Acc: {(100 * test_acc):>0.2f}%, Test NLL: {test_nll:>6f}")

    net.eval()

    num_classes = 10 
    all_noise, _ = compute_labelnoise(vis_loader, net, optim, device, num_classes, n_samples, args.bs, mc_samples)

    all_noise = [np.linalg.norm(x,2) for x in all_noise]

    #sqrt_var, dsig, sqrt_cov = compute_variance(net, optim, vis_loader,n_samples,device)

    #sqrt_var_norm = [x for xs in sqrt_var for x in xs]
    #dsig_norm = [y for ys in dsig for y in ys]
    #sqrt_cov_norm = [z for zs in sqrt_cov for z in zs]

    index=list(range(n_samples))
    labels = tr_targets

    with h5py.File(output_file, 'w') as f:
        f.create_dataset("images", data=torch.stack([ds_train[i][0] for i in index]).numpy())  # Save sorted images
        f.create_dataset("labels", data=np.array(labels))  # Sorted labels
        f.create_dataset("noise", data=np.array(all_noise))
        
    print(f"Saved MNIST images, labels, and noise values to {output_file}")

    sort_noises,index,labels=zip(*sorted(zip(all_noise,index,labels),reverse=True))

    linewidth = 5
    fontsize = 20

    ls_noise = [0.01] * num_classes
    ls_noise[0] = -0.09

    ls_noise = np.linalg.norm(ls_noise,2)

    fig, ax = plt.subplots(figsize=(8, 6))
    print(ls_noise)

    ax.plot(sort_noises, label = 'IVON',color='red', linewidth=linewidth )
    ax.plot([ls_noise for i in range(len(sort_noises))], label = 'Label Smoothing', linestyle='dashed', color='gray',linewidth=linewidth)

    # Remove the x-axis ticks
    plt.xticks([])
    plt.yticks([0.1, 0.3, 0.5],fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15, labelcolor='dimgray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('dimgray')
    ax.spines['left'].set_color('dimgray')

    # Optionally, you can also remove the x-axis label or customize it further
    plt.xlabel('Examples',fontsize=fontsize)
    plt.ylabel(r'Label Noise $\|\epsilon\|_2$',fontsize=fontsize)

    plt.legend(loc = 'upper right',fontsize=15)

    # Show the plot
    plt.savefig('./noise_distirbution.pdf')
    plt.savefig('./noise_distirbution.png',dpi=600)
    #plt.show()