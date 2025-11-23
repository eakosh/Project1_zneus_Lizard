import os
from typing import Optional, Tuple, Callable
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from dataset import LizardDataset

class LizardDataModule(pl.LightningDataModule):
    """
    DataModule for Lizard Dataset
    
    Handles:
    - Data loading and preprocessing
    - Train/val/test splits
    - Data augmentation
    - Class imbalance handling
    - Multi-source data handling
    """
    
    def __init__(
        self,
        data_root: str,
        target_size: Tuple[int, int] = (256, 256),
        batch_size: int = 16,
        num_workers: int = 4,
        preprocessing_fn: Optional[Callable] = None,
        use_weighted_sampler: bool = True,
        pin_memory: bool = True,
        return_instance_map: bool = True,
        return_classification: bool = True,
        custom_train_transform: Optional[Callable] = None,
        custom_val_transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_root: Root directory containing the dataset
            target_size: Target image size (H, W)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            preprocessing_fn: Preprocessing function from segmentation_models_pytorch
            use_weighted_sampler: Whether to use weighted random sampler for training
            pin_memory: Whether to use pinned memory
            return_instance_map: Whether to return instance segmentation maps
            return_classification: Whether to return classification labels
            custom_train_transform: Custom training augmentation pipeline
            custom_val_transform: Custom validation augmentation pipeline
        """
        super().__init__()
        self.data_root = data_root
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_fn = preprocessing_fn
        self.use_weighted_sampler = use_weighted_sampler
        self.pin_memory = pin_memory
        self.return_instance_map = return_instance_map
        self.return_classification = return_classification
        
        self.images_path_1 = os.path.join(data_root, 'lizard_images1', 'Lizard_Images1')
        self.images_path_2 = os.path.join(data_root, 'lizard_images2', 'Lizard_Images2')
        self.labels_path = os.path.join(data_root, 'lizard_labels', 'Lizard_Labels', 'Labels')
        self.info_csv = os.path.join(data_root, 'lizard_labels', 'Lizard_Labels', 'info.csv')
        
        if custom_train_transform is not None:
            self.train_transform = custom_train_transform
        else:
            self.train_transform = get_training_augmentation(target_size)
        
        if custom_val_transform is not None:
            self.val_transform = custom_val_transform
        else:
            self.val_transform = get_validation_augmentation(target_size)
        
        self.preprocessing = get_preprocessing(preprocessing_fn)
        
        self.train_dataset: Optional[LizardDataset] = None
        self.val_dataset: Optional[LizardDataset] = None
        self.test_dataset: Optional[LizardDataset] = None
        
        self.train_sampler_weights: Optional[torch.Tensor] = None
        
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (fit, validate, test, predict)
        Called on every GPU
        """
        if stage == 'fit' or stage is None:
            # Training dataset (split=1)
            self.train_dataset = LizardDataset(
                images_path_1=self.images_path_1,
                images_path_2=self.images_path_2,
                labels_path=self.labels_path,
                csv_file=self.info_csv,
                split=1,  # Train split
                target_size=self.target_size,
                transform=self.train_transform,
                preprocessing=self.preprocessing,
                return_instance_map=self.return_instance_map,
                return_classification=self.return_classification,
                normalize=False,  # Normalization is done in transform
            )
            
            # Validation dataset (split=2)
            self.val_dataset = LizardDataset(
                images_path_1=self.images_path_1,
                images_path_2=self.images_path_2,
                labels_path=self.labels_path,
                csv_file=self.info_csv,
                split=2,  # Validation split
                target_size=self.target_size,
                transform=self.val_transform,
                preprocessing=self.preprocessing,
                return_instance_map=self.return_instance_map,
                return_classification=self.return_classification,
                normalize=False,
            )
            
            # Compute weighted sampler weights if needed
            if self.use_weighted_sampler:
                self._compute_sampler_weights()
        
        if stage == 'test' or stage is None:
            # Test dataset (split=3)
            self.test_dataset = LizardDataset(
                images_path_1=self.images_path_1,
                images_path_2=self.images_path_2,
                labels_path=self.labels_path,
                csv_file=self.info_csv,
                split=3,  # Test split
                target_size=self.target_size,
                transform=self.val_transform,
                preprocessing=self.preprocessing,
                return_instance_map=self.return_instance_map,
                return_classification=self.return_classification,
                normalize=False,
            )
    
    def _compute_sampler_weights(self):
        """
        Compute weights for WeightedRandomSampler based on class distribution
        
        This helps balance the training by sampling underrepresented classes more often
        """
        if self.train_dataset is None:
            return
        
        # Count samples per source (since sources have different characteristics)
        import pandas as pd
        info_df = pd.read_csv(self.info_csv)
        info_df['source_prefix'] = info_df['Filename'].str.split('_').str[0]
        train_df = info_df[info_df['Split'] == 1]
        
        source_counts = train_df['source_prefix'].value_counts()
        total_samples = len(train_df)
        
        # Compute inverse frequency weights for each source
        source_weights = {}
        for source in source_counts.index:
            source_weights[source] = total_samples / (len(source_counts) * source_counts[source])
        
        # Assign weight to each sample based on its source
        sample_weights = []
        for idx in range(len(self.train_dataset)):
            source = self.train_dataset.data.iloc[idx]['source_prefix']
            sample_weights.append(source_weights[source])
        
        self.train_sampler_weights = torch.DoubleTensor(sample_weights)
        
        print("\nWeighted Sampler Statistics:")
        print(f"Total training samples: {total_samples}")
        print(f"Source distribution:")
        for source, count in source_counts.items():
            weight = source_weights[source]
            print(f"  {source}: {count} samples (weight: {weight:.3f})")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if self.use_weighted_sampler and self.train_sampler_weights is not None:
            sampler = WeightedRandomSampler(
                weights=self.train_sampler_weights,
                num_samples=len(self.train_sampler_weights),
                replacement=True
            )
            shuffle = False  # Don't shuffle when using sampler
        else:
            sampler = None
            shuffle = True
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Drop last incomplete batch
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    


# Example usage
if __name__ == "__main__":
    # Initialize datamodule
    datamodule = LizardDataModule(
        data_root='./data',
        target_size=(256, 256),
        batch_size=8,
        num_workers=4,
        use_weighted_sampler=True,
    )
    
    # Setup datasets
    datamodule.setup('fit')
    
    # Print statistics
    datamodule.print_statistics()
    
    # Test dataloader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    print("\nSample Batch:")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Mask shape: {batch['mask'].shape}")
    if 'inst_map' in batch:
        print(f"  Instance map shape: {batch['inst_map'].shape}")
    print(f"  Sources: {batch['source']}")
    print(f"  Nuclei counts: {batch['nuclei_count']}")



def get_training_augmentation(target_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """
    Get training augmentation pipeline
    Geometric transformations
    Color augmentations (mild for H&E stained images)
    Noise and blur
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),
        
        # Elastic deformations 
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.3
        ),
        
        # Color augmentations
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
        ], p=0.5),
        
        # Brightness and contrast
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Normalization and conversion to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_augmentation(target_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Get validation/test augmentation pipeline (only normalization)"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_preprocessing(preprocessing_fn: Optional[Callable] = None) -> Callable:
    """
    Get preprocessing function for specific encoder
    
    Args:
        preprocessing_fn: Preprocessing function from segmentation_models_pytorch
        
    Returns:
        Preprocessing transform
    """
    if preprocessing_fn is None:
        return lambda x: x
    
    return A.Lambda(image=preprocessing_fn)
