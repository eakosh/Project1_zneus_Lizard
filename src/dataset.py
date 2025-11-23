import os
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd


class LizardDataset(Dataset):
    """
    Lizard Dataset for Nuclear Instance Segmentation and Classification
    """
    def __init__(
        self,
        images_path_1: str,
        images_path_2: str,
        labels_path: str,
        csv_file: str,
        split: int = 1,  # 1=train, 2=val, 3=test
        target_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        return_instance_map: bool = True,
        return_classification: bool = True,
        normalize: bool = True,
    ):
        """
        Args:
            images_path_1: Path to first images folder (Lizard_Images1)
            images_path_2: Path to second images folder (Lizard_Images2)
            labels_path: Path to labels folder
            csv_file: Path to info.csv file
            split: Dataset split (1=train, 2=val, 3=test)
            target_size: Target image size (H, W)
            transform: Albumentations transform for augmentation
            preprocessing: Additional preprocessing function
            return_instance_map: Whether to return instance segmentation map
            return_classification: Whether to return class labels
            normalize: Whether to normalize images to [0, 1]
        """
        self.images_path_1 = images_path_1
        self.images_path_2 = images_path_2
        self.labels_path = labels_path
        self.target_size = target_size
        self.transform = transform
        self.preprocessing = preprocessing
        self.return_instance_map = return_instance_map
        self.return_classification = return_classification
        self.normalize = normalize
        
        self.info_df = pd.read_csv(csv_file)
        self.info_df['source_prefix'] = self.info_df['Filename'].str.split('_').str[0]
        
        self.data = self.info_df[self.info_df['Split'] == split].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def load_image(self, filename: str) -> Optional[np.ndarray]:
        """Load image from either images path"""
        img_path = os.path.join(self.images_path_1, filename)
        if os.path.exists(img_path):
            return np.array(Image.open(img_path))
        
        img_path = os.path.join(self.images_path_2, filename)
        if os.path.exists(img_path):
            return np.array(Image.open(img_path))
        
        return None
    
    def load_label(self, label_path: str) -> Optional[Dict]:
        """Load .mat label file"""
        try:
            label = sio.loadmat(label_path)
            return {
                'inst_map': label['inst_map'],
                'id': label['id'],
                'class': label['class'],
                'bbox': label['bbox'],
                'centroid': label['centroid']
            }
        except Exception as e:
            print(f"Error loading {label_path}: {e}")
            return None
    
    def create_semantic_mask(self, inst_map: np.ndarray, classes: np.ndarray, 
                             nuclei_ids: np.ndarray) -> np.ndarray:
        """
        Create semantic segmentation mask from instance map and classes
        
        Args:
            inst_map: Instance segmentation map (H, W)
            classes: Class for each nucleus (N,)
            nuclei_ids: ID for each nucleus (N,)
            
        Returns:
            Semantic mask (H, W) with class labels
        """
        semantic_mask = np.zeros_like(inst_map, dtype=np.int64)
        
        for nuc_id, nuc_class in zip(nuclei_ids, classes):
            semantic_mask[inst_map == nuc_id] = nuc_class
        
        return semantic_mask
    
    def resize_with_padding(self, image: np.ndarray, mask: np.ndarray, 
                            inst_map: Optional[np.ndarray] = None) -> Tuple:
        """
        Resize image and masks with padding to maintain aspect ratio
        
        Args:
            image: Input image (H, W, C)
            mask: Semantic mask (H, W)
            inst_map: Instance map (H, W), optional
            
        Returns:
            Resized and padded image, mask, and optionally inst_map
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        image_padded = cv2.copyMakeBorder(
            image_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        mask_padded = cv2.copyMakeBorder(
            mask_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
        
        if inst_map is not None:
            inst_map_resized = cv2.resize(inst_map, (new_w, new_h), 
                                         interpolation=cv2.INTER_NEAREST)
            inst_map_padded = cv2.copyMakeBorder(
                inst_map_resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=0
            )
            return image_padded, mask_padded, inst_map_padded
        
        return image_padded, mask_padded
    
    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray,
                            inst_map: Optional[np.ndarray] = None) -> Tuple:
        """Apply albumentations transforms"""
        if self.transform is None:
            return image, mask, inst_map
        
        if inst_map is not None:
            transformed = self.transform(image=image, masks=[mask, inst_map])
            image = transformed['image']
            mask, inst_map = transformed['masks']
            return image, mask, inst_map
        else:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            return image, mask, None
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Returns:
            - image: Tensor of shape (C, H, W)
            - mask: Semantic segmentation mask (H, W)
            - inst_map: Instance segmentation map (H, W), optional
            - filename: Image filename
            - source: Data source (consep, crag, dpath, glas, pannuke)
            - nuclei_count: Number of nuclei in image
            - original_size: Original image size (H, W)
        """
        row = self.data.iloc[idx]
        filename = row['Filename']
        source = row['source_prefix']
        
        img_name = f"{filename}.png"
        image = self._load_image(img_name)
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        
        original_size = image.shape[:2]
        
        label_name = f"{filename}.mat"
        label_path = os.path.join(self.labels_path, label_name)
        label_data = self._load_label(label_path)
        
        if label_data is None:
            raise FileNotFoundError(f"Label not found: {label_name}")
        
        inst_map = label_data['inst_map'].astype(np.int32)
        classes = label_data['class'].flatten().astype(np.int64)
        nuclei_ids = label_data['id'].flatten().astype(np.int32)
        
        semantic_mask = self._create_semantic_mask(inst_map, classes, nuclei_ids)
        
        if self.return_instance_map:
            image, semantic_mask, inst_map = self._resize_with_padding(
                image, semantic_mask, inst_map
            )
        else:
            image, semantic_mask = self._resize_with_padding(
                image, semantic_mask
            )
            inst_map = None
        
        image, semantic_mask, inst_map = self._apply_augmentations(
            image, semantic_mask, inst_map
        )
        
        if self.normalize and not isinstance(image, torch.Tensor):
            image = image.astype(np.float32) / 255.0
        
        if self.preprocessing:
            image = self.preprocessing(image)
        
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if not isinstance(semantic_mask, torch.Tensor):
            semantic_mask = torch.from_numpy(semantic_mask).long()
        
        output = {
            'image': image,
            'mask': semantic_mask,
            'filename': filename,
            'source': source,
            'nuclei_count': len(nuclei_ids),
            'original_size': original_size,
        }
        
        if self.return_instance_map and inst_map is not None:
            if not isinstance(inst_map, torch.Tensor):
                inst_map = torch.from_numpy(inst_map).long()
            output['inst_map'] = inst_map
        
        if self.return_classification:
            class_counts = torch.zeros(7, dtype=torch.long)  # 0=background + 6 classes
            for c in classes:
                class_counts[c] += 1
            output['class_counts'] = class_counts
        
        return output
