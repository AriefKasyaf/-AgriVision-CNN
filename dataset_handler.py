"""
dataset_handler.py
Modul untuk memuat turun dan menyediakan dataset (CIFAR-10) menggunakan
torchvision.datasets, serta memaparkan (plot) 3 gambar pertama beserta label.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


# Nama kelas dalam CIFAR-10 (untuk paparan label)
CLASS_NAMES = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def get_dataset(batch_size=32, data_dir='./data'):
    """
    Memuat turun dataset CIFAR-10 (train & test) dan mengembalikan DataLoader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def show_sample_images(dataset, num_images=3):
    """
    Memaparkan (plot) 'num_images' gambar pertama dari dataset beserta labelnya
    menggunakan Matplotlib dan NumPy.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(9, 3))

    for i in range(num_images):
        image, label = dataset[i]

        # Tensor (C,H,W) -> Numpy (H,W,C) untuk paparan, dan "unnormalize"
        img_np = image.numpy().transpose((1, 2, 0))
        img_np = img_np * 0.5 + 0.5  # unnormalize balik ke [0,1]
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        axes[i].set_title(CLASS_NAMES[label])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png')  # simpan untuk screenshot laporan
    plt.show()


if __name__ == "__main__":
    train_loader, test_loader = get_dataset(batch_size=32)
    train_set = train_loader.dataset
    show_sample_images(train_set, num_images=3)
    print("Dataset berjaya dimuat turun dan 3 gambar sampel dipaparkan.")
