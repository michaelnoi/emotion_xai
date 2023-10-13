import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

class FaceDataset(Dataset):
    def __init__(self, annotations_file, distribution_file, img_dir, transform=None, split='train'):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.distribution = pd.read_csv(distribution_file, header=None)

        if split == 'train' or split == 'val':
            train_idx, val_idx = train_test_split(range(len(self.img_labels)), train_size=0.8, random_state=42)
            if split == 'train':
                self.img_labels = self.img_labels.iloc[train_idx]
                self.distribution = self.distribution.iloc[train_idx]
            elif split == 'val':
                self.img_labels = self.img_labels.iloc[val_idx]
                self.distribution = self.distribution.iloc[val_idx]
        elif split != 'test':
            raise ValueError(f"Invalid split: {split}")
        
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0].split('.')[0]}_aligned.{self.img_labels.iloc[idx, 0].split('.')[1]}")
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        distribution = self.distribution.iloc[idx, 1:].values.astype(float)
        if self.transform:
            image = self.transform(image)
        return image, label, distribution
