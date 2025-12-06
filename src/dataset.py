import os
import glob
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Categories you are using
CATEGORIES = ["bottle", "hazelnut", "cable", "tile"]


class MvtecTrainDataset(Dataset):
    """
    TRAIN ONLY dataset:
    - loads train/good images for the selected categories
    - returns ONLY image tensors of shape [3, H, W]
    """

    def __init__(  # FIXED: Changed _init_ to __init__
        self,
        root_dir: str,
        categories: List[str],
        img_size: int = 256,
        transform=None,
    ):
        self.root_dir = root_dir
        self.categories = categories
        self.img_size = img_size

        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),  # [0, 1]
            ])
        else:
            self.transform = transform

        self.samples = self._gather_samples()
        print(f"Loaded {len(self.samples)} training images")

    def _gather_samples(self):
        samples = []
        for cat in self.categories:
            cat_dir = os.path.join(self.root_dir, cat)
            img_dir = os.path.join(cat_dir, "train", "good")
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist")
                continue
                
            img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            print(f"  {cat}: {len(img_paths)} images")
            for p in img_paths:
                samples.append(p)
        return samples

    def __len__(self):  # FIXED: Changed _len_ to __len__
        return len(self.samples)

    def __getitem__(self, idx: int):  # FIXED: Changed _getitem_ to __getitem__
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # In case of any weird file issue, return a black image instead of None
            img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        return img  # pure tensor


class MvtecTestDataset(Dataset):
    """
    TEST dataset:
    - loads all test images with labels
    - returns (image, label, category, defect_type, image_path)
    """
    
    def __init__(
        self,
        root_dir: str,
        categories: List[str],
        img_size: int = 256,
        transform=None,
    ):
        self.root_dir = root_dir
        self.categories = categories
        self.img_size = img_size
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])
        else:
            self.transform = transform
            
        self.samples = self._gather_samples()
        print(f"Loaded {len(self.samples)} test images")
        
    def _gather_samples(self):
        samples = []
        for cat in self.categories:
            cat_dir = os.path.join(self.root_dir, cat, "test")
            if not os.path.exists(cat_dir):
                continue
                
            for defect_type in sorted(os.listdir(cat_dir)):
                defect_dir = os.path.join(cat_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                    
                label = 0 if defect_type == "good" else 1
                img_paths = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
                for p in img_paths:
                    samples.append({
                        'path': p,
                        'label': label,
                        'category': cat,
                        'defect_type': defect_type
                    })
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample['path']).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            
        return img, sample['label'], sample['category'], sample['defect_type'], sample['path']


def get_dataloaders(
    root_dir: str,
    categories: List[str] = None,
    img_size: int = 256,
    batch_size: int = 16,
    train_only: bool = True,
):
    """
    Returns:
      train_loader, test_loader (or None if train_only=True)
    """
    if categories is None:
        categories = CATEGORIES

    train_dataset = MvtecTrainDataset(
        root_dir=root_dir,
        categories=categories,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )
    
    test_loader = None
    if not train_only:
        test_dataset = MvtecTestDataset(
            root_dir=root_dir,
            categories=categories,
            img_size=img_size,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    return train_loader, test_loader
