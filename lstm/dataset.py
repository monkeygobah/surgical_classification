from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms


def create_data_loaders(dataset, batch_size=32, num_workers=4, indices=None, train=False, phase_exps=False):
    if indices is None:
        indices = list(range(len(dataset)))

    if train:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset.transform = train_transforms
        
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset.transform = train_transforms

    if not phase_exps:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    return data_loader



