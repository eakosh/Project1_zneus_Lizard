import os
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class PatchDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root_dir, split, "img")
        self.mask_dir = os.path.join(root_dir, split, "mask")

        self.img_paths = sorted(glob(os.path.join(self.img_dir, "*.png")))
        assert len(self.img_paths) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.basename(img_path)

        mask_path = os.path.join(self.mask_dir, name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  

        img = np.array(img)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8) 

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.long)

        return img, mask
