import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from patch_dataset import PatchDataset


class PatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir="patches",
        batch_size=8,
        num_workers=4,
        img_size=512,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        self.train_ds = PatchDataset(
            root_dir=self.root_dir,
            split="train",
            transform=self.train_transform,
        )

        self.val_ds = PatchDataset(
            root_dir=self.root_dir,
            split="val",
            transform=self.val_transform,
        )

        self.test_ds = PatchDataset(
            root_dir=self.root_dir,
            split="test",
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
