import random
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from torchvision import datasets, transforms

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

train_full = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)

test_set = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

print("Official training samples:", len(train_full))
print("Official test samples:", len(test_set))
train_set, validation_set = random_split(
    train_full,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42),
)

print("Training samples:", len(train_set))
print("Validation samples:", len(validation_set))

train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)

validation_loader = DataLoader(validation_set, batch_size=128)
test_loader = DataLoader(test_set, batch_size=128)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)