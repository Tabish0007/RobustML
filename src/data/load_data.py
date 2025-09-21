import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data(batch_size=64):
    # Example: CIFAR-10, replace with your dataset later
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
