import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

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

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0, mask_value=0, p=0.5
            ),

            A.OneOf([
                A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=1.0),
                A.HueSaturationValue(10, 15, 10, p=1.0)
            ], p=0.4),

            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                A.GaussianBlur(3, p=1.0),
                A.MedianBlur(3, p=1.0),
            ], p=0.3),

            A.OneOf([
                A.ElasticTransform(alpha=20, sigma=20, alpha_affine=10,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0, mask_value=0, p=1.0),
                A.GridDistortion(distort_limit=0.05,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0, mask_value=0, p=1.0),
            ], p=0.2),

            A.OneOf([
                A.RGBShift(10, 10, 10, p=1.0),
                A.RandomBrightnessContrast(0.15, 0.15, p=1.0),
            ], p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
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
