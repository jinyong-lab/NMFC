# Last updated: 2026-04-15 18:30
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────
# Supported datasets
# ─────────────────────────────────────────────────────────────

DATASET_CLASSES = {
    "cifar10": ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'],
    "fashionmnist": ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    "stl10": ['airplane', 'bird', 'car', 'cat', 'deer',
              'dog', 'horse', 'monkey', 'ship', 'truck'],
}

# EN: Keep legacy name for backward compatibility with diagnostics.py / experiment.ipynb
# KR: diagnostics.py / experiment.ipynb 하위 호환성을 위해 기존 이름 유지
CIFAR10_CLASSES = DATASET_CLASSES["cifar10"]


def get_loaders(dataset_name, data_dir='./data', batch_size=128, num_workers=2):
    """
    Return (train_loader, test_loader) for the requested dataset.

    Args:
        dataset_name : "cifar10", "fashionmnist", or "stl10"
        data_dir     : path to save/load raw data
        batch_size   : samples per mini-batch
        num_workers  : DataLoader worker processes

    EN: Fashion-MNIST images are 28×28 grayscale. They are converted to
        3-channel RGB by repeating the single channel (Grayscale(3)) so
        the frozen ResNet-18 backbone (which expects 3-channel input)
        works without modification.
        STL-10 images are 96×96 RGB — resized to 224 like CIFAR-10.
        STL-10 uses split='train'/'test' instead of train=True/False.
        Train set: 5,000 images. Test set: 8,000 images.
    KR: Fashion-MNIST 이미지는 28×28 흑백. 단일 채널을 3채널 RGB로
        복제(Grayscale(3))하여 3채널 입력을 요구하는 고정된 ResNet-18
        백본을 수정 없이 사용할 수 있게 함.
        STL-10 이미지는 96×96 RGB — CIFAR-10과 동일하게 224로 리사이즈.
        STL-10은 train=True/False 대신 split='train'/'test' 사용.
        훈련 세트: 5,000장. 테스트 세트: 8,000장.
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_transform)
        test_dataset  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "fashionmnist":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True,  download=True, transform=train_transform)
        test_dataset  = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "stl10":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                                 std=[0.2603, 0.2566, 0.2713])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                                 std=[0.2603, 0.2566, 0.2713])
        ])
        train_dataset = datasets.STL10(root=data_dir, split='train', download=True, transform=train_transform)
        test_dataset  = datasets.STL10(root=data_dir, split='test',  download=True, transform=test_transform)

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose 'cifar10', 'fashionmnist', or 'stl10'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# v4 (2026-04-20): Return BOTH augmented and clean train datasets.
# Rationale (Ref 5 SimCLR §3): val/test must use clean transforms.
# Returning both lets make_train_val_loaders split by index and assign
# augmented transform to train subset, clean transform to val subset.
def get_train_datasets_both_transforms(dataset_name, data_dir='./data'):
    """Return (train_ds_augmented, train_ds_clean, test_loader)."""
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
        clean = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
        train_aug = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=aug)
        train_clean = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=clean)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=clean)

    elif dataset_name == "fashionmnist":
        aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        clean = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        train_aug = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=aug)
        train_clean = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=clean)
        test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=clean)

    elif dataset_name == "stl10":
        aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                                 std=[0.2603, 0.2566, 0.2713])
        ])
        clean = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                                 std=[0.2603, 0.2566, 0.2713])
        ])
        train_aug = datasets.STL10(root=data_dir, split='train', download=True, transform=aug)
        train_clean = datasets.STL10(root=data_dir, split='train', download=True, transform=clean)
        test_ds = datasets.STL10(root=data_dir, split='test', download=True, transform=clean)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return train_aug, train_clean, test_ds


# EN: Legacy wrapper kept for backward compatibility with diagnostics.py
# KR: diagnostics.py 하위 호환성을 위해 기존 래퍼 유지
def get_cifar10_loaders(data_dir='./data', batch_size=128, num_workers=2):
    return get_loaders("cifar10", data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
