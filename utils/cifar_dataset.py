import torch
from torchvision import datasets, transforms

def get_cifar_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train = datasets.CIFAR10(root="data/cifar", train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root="data/cifar", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
