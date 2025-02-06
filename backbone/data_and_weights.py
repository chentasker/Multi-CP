import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from medmnist import INFO
import medmnist
import torch.nn as nn
from load_config import *
config = load_config()

def load_data(current_config, train_size_p=0.65, val_size_p=0.1, test_size_p=0.1, cal_size_p=0.15, data_folder='./data'):
    torch.manual_seed(5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if config['general']['dataset_name'] == "CIFAR100":
            train_set = datasets.CIFAR100(root=data_folder, train=True, download=True, transform=transform)
            test_set = datasets.CIFAR100(root=data_folder, train=False, download=True, transform=transform)
            dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    elif config['general']['dataset_name'] == "PathMNIST":
        info = INFO["pathmnist"]
        DataClass = getattr(medmnist, info['python_class'])
        train_set = DataClass(root=data_folder, split='train', transform=transform, download=True, as_rgb=True)
        test_set = DataClass(root=data_folder, split='test', transform=transform, download=True, as_rgb=True)
        val_set = DataClass(root=data_folder, split='val', transform=transform, download=True, as_rgb=True)
        val_set.labels = np.squeeze(val_set.labels, 1)
        train_set.labels = np.squeeze(train_set.labels, 1)
        test_set.labels = np.squeeze(test_set.labels, 1)
        dataset = ConcatDataset([train_set, test_set, val_set])
    elif config['general']['dataset_name'] == "MNIST":
        transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform_mnist)
        test_set = datasets.MNIST(root=data_folder, train=False, download=True, transform=transform_mnist)
        dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    else:
        raise ValueError("Unknown dataset name")

    # Split the dataset into training, validation, test, and calibration sets
    num_samples = len(dataset)
    train_size = int(train_size_p * num_samples)
    val_size = int(val_size_p * num_samples)
    test_size = int(test_size_p * num_samples)
    cal_size = int(cal_size_p * num_samples)

    remaining_size = num_samples - train_size - val_size - test_size - cal_size

    train_set, val_set, test_set, cal_set, remaining_set = random_split(
        dataset, [train_size, val_size, test_size, cal_size, remaining_size]
    )
    train_set = torch.utils.data.ConcatDataset([train_set, remaining_set])

    # Create loaders for training, validation, test, and calibration sets
    train_loader = DataLoader(train_set, batch_size=current_config['current_step']['batch_size'], shuffle=True,
                              num_workers=1)
    val_loader = DataLoader(val_set, batch_size=current_config['current_step']['batch_size'], shuffle=False,
                            num_workers=1)
    test_loader = DataLoader(test_set, batch_size=current_config['current_step']['batch_size'], shuffle=False,
                             num_workers=1)
    cal_loader = DataLoader(cal_set, batch_size=current_config['current_step']['batch_size'], shuffle=False,
                            num_workers=1)

    return train_loader, val_loader, test_loader, cal_loader


def save_best_weights(model, optimizer, epoch, val_loss, save_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(state, save_path)
    print(f"Best weights saved to {save_path}")


# Assuming 'model' is an instance of ModifiedResNet50
def load_best_weights(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Loaded best weights from {load_path}, Epoch: {epoch}, Validation Loss: {val_loss}")
    return model, val_loss


def print_layer_names_and_sizes(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            print(f"{name}: {layer.weight.size()}")
