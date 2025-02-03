import torchvision
import torchvision.transforms as transforms
import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


transform_usps = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2469,), (0.2989,)),
])
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
transform_fmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_moon = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
])
transform_tSNE_mnist = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
])

def get_dataset(name_dataset, return_transform=False, n_samples=1000, noise=5, test_split=0.2):
    if name_dataset == 'MNIST':
        ds_train, ds_test = load_mnist()
        transform = transform_mnist
    elif name_dataset == 'FMNIST':
        ds_train, ds_test = load_fmnist()
        transform = transform_fmnist
    elif name_dataset == 'CIFAR10':
        ds_train, ds_test = load_cifar10()
        transform = transform_cifar10
    elif name_dataset == 'MOON':
        ds_train, ds_test = load_moon(n_samples, noise, test_split)
        transform = transform_moon
    elif name_dataset == 'MNIST_REDUX':
        ds_train, ds_test = load_tSNE_mnist()
        transform = transform_tSNE_mnist
    else:
        raise NotImplementedError
    if return_transform:
        return ds_train, ds_test, transform
    else:
        return ds_train, ds_test

def load_cifar10():
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_cifar10)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=transform_cifar10)
    testset.targets = torch.asarray(testset.targets)
    return trainset, testset

def load_usps():
    trainset = torchvision.datasets.USPS(
        root='../data', train=True, download=True, transform=transform_usps)
    testset = torchvision.datasets.USPS(
        root='../data', train=False, download=True, transform=transform_usps)
    return trainset, testset

def load_mnist():
    trainset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_mnist)
    testset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform_mnist)
    return trainset, testset

def load_tSNE_mnist():
    trainset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_mnist)
    testset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform_mnist)

    # Convert datasets to NumPy arrays
    train_images = np.array([trainset[i][0].numpy().reshape(-1) for i in range(len(trainset))])
    train_labels = np.array([trainset[i][1] for i in range(len(trainset))])
    
    test_images = np.array([testset[i][0].numpy().reshape(-1) for i in range(len(testset))])
    test_labels = np.array([testset[i][1] for i in range(len(testset))])

    # Step 1: PCA (reduce to 50 dimensions)
    pca = PCA(n_components=50)
    train_pca = pca.fit_transform(train_images)
    test_pca = pca.transform(test_images)

    # Step 2: t-SNE (reduce to 2 dimensions)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    train_tsne = tsne.fit_transform(train_pca)
    test_tsne = tsne.fit_transform(test_pca)

    X_train = torch.tensor(train_tsne, dtype=torch.float32)
    X_test = torch.tensor(test_tsne, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    # Create a custom Dataset
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    return trainset, testset



def load_fmnist():
    trainset = torchvision.datasets.FashionMNIST(
        root='../data', train=True, download=True, transform=transform_fmnist)
    testset = torchvision.datasets.FashionMNIST(
        root='../data', train=False, download=True, transform=transform_fmnist)
    return trainset, testset

def load_moon(n_samples=1000, noise=0.2, test_split=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # Split into train/test
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create a custom Dataset
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)

    return trainset, testset