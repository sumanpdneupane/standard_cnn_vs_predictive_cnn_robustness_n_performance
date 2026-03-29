import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def download():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data/', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transform)
    return train_dataset, test_dataset