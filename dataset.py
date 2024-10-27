# dataset.py

from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class SalientObjectDataset(Dataset):
    def __init__(self, image_root, mask_root, edge_root=None, transform=None):
        self.image_paths = sorted([os.path.join(image_root, img) for img in os.listdir(image_root)])
        self.mask_paths = sorted([os.path.join(mask_root, img) for img in os.listdir(mask_root)])
        self.edge_paths = sorted([os.path.join(edge_root, img) for img in os.listdir(edge_root)]) if edge_root else None
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = {'image': image, 'mask': mask}

        if self.edge_paths:
            edge = Image.open(self.edge_paths[idx]).convert('L')
            edge = self.mask_transform(edge)
            sample['edge'] = edge

        return sample
