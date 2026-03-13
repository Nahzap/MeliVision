"""
Enhanced Data Transforms with Microscopy-Specific Augmentations
Based on SOTA research (2024) for microscopy image analysis

Author: MetricLearning Project
Date: 2026-01-05
"""

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class MicroscopyAugmentation:
    """
    Microscopy-specific augmentation pipeline using Albumentations.
    Designed for pollen/honey microscopy images with domain-specific transforms.
    """
    
    def __init__(self, config=None):
        """
        Initialize microscopy augmentation pipeline.
        
        Args:
            config (dict): Configuration dictionary with augmentation parameters
        """
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        self.transform = self._build_transform()
    
    def _get_default_config(self):
        """Default configuration for microscopy augmentation"""
        return {
            'image_size': 224,
            'resize_size': 256,
            'color_jitter': {
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.2,
                'hue': 0.1
            },
            'microscopy_specific': {
                'random_gamma': {'enabled': True, 'gamma_limit': [80, 120], 'p': 0.5},
                'gaussian_noise': {'enabled': True, 'var_limit': [10.0, 50.0], 'p': 0.5},
                'blur': {'enabled': True, 'blur_limit': 3, 'p': 0.3},
                'optical_distortion': {'enabled': True, 'distort_limit': 0.1, 'p': 0.3},
                'clahe': {'enabled': True, 'clip_limit': 2.0, 'tile_grid_size': [8, 8], 'p': 0.5}
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    def _build_transform(self):
        """Build Albumentations transform pipeline — optimized for DINOv2 fine-tuning."""
        transform_list = []
        image_size = self.config['image_size']
        resize_size = self.config.get('resize_size', image_size)
        
        # Scale augmentation: RandomResizedCrop gives multi-scale views of the grain
        # (SOTA for fine-grained recognition — Touvron et al., DeiT III, ECCV 2022)
        if resize_size > image_size:
            transform_list.append(A.Resize(resize_size, resize_size))
            transform_list.append(A.RandomCrop(image_size, image_size))
        else:
            transform_list.append(
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5,
                )
            )
            # Fallback resize when RandomResizedCrop doesn't fire (p=0.5)
            transform_list.append(A.Resize(image_size, image_size))
        
        # Geometric transforms (appropriate for microscopy — grains have no canonical orientation)
        transform_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.8, border_mode=0, fill=128),
            A.Affine(
                translate_percent={'x': (-0.08, 0.08), 'y': (-0.08, 0.08)},
                scale=(0.9, 1.1),
                rotate=0,
                border_mode=0, fill=128,
                p=0.4
            )
        ])
        
        # Color augmentations — ONE unified block (no redundancy)
        cj = self.config['color_jitter']
        transform_list.append(
            A.ColorJitter(
                brightness=cj['brightness'],
                contrast=cj['contrast'],
                saturation=cj['saturation'],
                hue=cj['hue'],
                p=0.8
            )
        )
        
        # Microscopy-specific augmentations (SOTA 2024)
        micro = self.config['microscopy_specific']
        
        if micro['random_gamma']['enabled']:
            transform_list.append(
                A.RandomGamma(
                    gamma_limit=tuple(micro['random_gamma']['gamma_limit']),
                    p=micro['random_gamma']['p']
                )
            )
        
        if micro['gaussian_noise']['enabled']:
            transform_list.append(
                A.GaussNoise(
                    std_range=tuple(v / 255.0 for v in micro['gaussian_noise']['var_limit']),
                    p=micro['gaussian_noise']['p']
                )
            )
        
        if micro['blur']['enabled']:
            transform_list.append(
                A.Blur(
                    blur_limit=micro['blur']['blur_limit'],
                    p=micro['blur']['p']
                )
            )
        
        if micro['optical_distortion']['enabled']:
            transform_list.append(
                A.OpticalDistortion(
                    distort_limit=micro['optical_distortion']['distort_limit'],
                    p=micro['optical_distortion']['p']
                )
            )
        
        if micro['clahe']['enabled']:
            transform_list.append(
                A.CLAHE(
                    clip_limit=micro['clahe']['clip_limit'],
                    tile_grid_size=tuple(micro['clahe']['tile_grid_size']),
                    p=micro['clahe']['p']
                )
            )
        
        # Domain-robustness augmentations (simulate different microscopes/preparations)
        # NOTE: removed redundant RandomBrightnessContrast (already covered by ColorJitter above)
        domain_aug = self.config.get('domain_augmentation', {})
        if domain_aug.get('enabled', True):
            rgb_shift = domain_aug.get('rgb_shift_limit', 20)
            tone_scale = domain_aug.get('tone_curve_scale', 0.15)
            ds_min = domain_aug.get('downscale_min', 0.6)
            cd_holes = domain_aug.get('coarse_dropout_holes', 3)
            transform_list.extend([
                A.RGBShift(
                    r_shift_limit=rgb_shift, g_shift_limit=rgb_shift,
                    b_shift_limit=rgb_shift, p=0.3
                ),
                A.RandomToneCurve(scale=tone_scale, p=0.2),
                A.Downscale(
                    scale_range=(ds_min, 0.8), p=0.15
                ),
                A.CoarseDropout(
                    num_holes_range=(1, cd_holes), hole_height_range=(8, 20),
                    hole_width_range=(8, 20), fill=128, p=0.15
                ),
            ])
        
        # Normalization and tensor conversion
        transform_list.extend([
            A.Normalize(
                mean=self.config['normalize']['mean'],
                std=self.config['normalize']['std']
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transform_list)
    
    def __call__(self, image):
        """
        Apply augmentation to image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Augmented image tensor
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        augmented = self.transform(image=image)
        return augmented['image']


def get_train_transforms(config=None, mode='full'):
    """
    Get training transforms with microscopy-specific augmentations.
    
    Args:
        config (dict, optional): Configuration dictionary
        mode (str): 'full' for microscopy augmentations, 'deit3' for DeiT III 3-Augment
        
    Returns:
        MicroscopyAugmentation or DeiT3Augmentation or transforms.Compose
    """
    if mode == 'deit3':
        return DeiT3Augmentation(config)
    
    try:
        # Try to use enhanced microscopy augmentation
        return MicroscopyAugmentation(config)
    except ImportError:
        # Fallback to torchvision transforms if albumentations not available
        print("Warning: albumentations not available, using basic torchvision transforms")
        return _get_basic_train_transforms(config)


def _get_basic_train_transforms(config=None):
    """
    Fallback training transforms using torchvision only.
    
    Args:
        config (dict, optional): Configuration dictionary
        
    Returns:
        transforms.Compose: Transform pipeline
    """
    if config is None:
        config = {
            'resize_size': 256,
            'image_size': 224,
            'random_horizontal_flip': 0.5,
            'random_vertical_flip': 0.5,
            'random_rotation': 180,
            'color_jitter': {
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.2,
                'hue': 0.1
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    transform_list = [
        transforms.Resize((config['resize_size'], config['resize_size'])),
        transforms.RandomCrop(config['image_size']),
        transforms.RandomHorizontalFlip(p=config['random_horizontal_flip']),
        transforms.RandomVerticalFlip(p=config['random_vertical_flip']),
        transforms.RandomRotation(degrees=config['random_rotation']),
        transforms.ColorJitter(
            brightness=config['color_jitter']['brightness'],
            contrast=config['color_jitter']['contrast'],
            saturation=config['color_jitter']['saturation'],
            hue=config['color_jitter']['hue']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['normalize']['mean'],
            std=config['normalize']['std']
        )
    ]
    
    return transforms.Compose(transform_list)


class ValidationTransform:
    """Wrapper for validation transforms using Albumentations"""
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'image_size': 224,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        
        self.transform = A.Compose([
            A.Resize(config['image_size'], config['image_size']),
            A.Normalize(
                mean=config['normalize']['mean'],
                std=config['normalize']['std']
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        """Apply validation transform to image"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        augmented = self.transform(image=image)
        return augmented['image']


def get_val_transforms(config=None):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        config (dict, optional): Configuration dictionary
        
    Returns:
        ValidationTransform or transforms.Compose: Transform pipeline
    """
    if config is None:
        config = {
            'image_size': 224,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    try:
        # Try Albumentations for consistency
        return ValidationTransform(config)
    except ImportError:
        # Fallback to torchvision
        return transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize']['mean'],
                std=config['normalize']['std']
            )
        ])


def get_test_time_augmentation(config=None, n_augmentations=5):
    """
    Get Test-Time Augmentation (TTA) transforms for ensemble predictions.
    
    Args:
        config (dict, optional): Configuration dictionary
        n_augmentations (int): Number of augmented versions per image
        
    Returns:
        list: List of transform pipelines
    """
    if config is None:
        config = {
            'image_size': 224,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    tta_transforms = []
    
    # Original (no augmentation)
    tta_transforms.append(get_val_transforms(config))
    
    # Horizontal flip
    tta_transforms.append(
        transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize']['mean'],
                std=config['normalize']['std']
            )
        ])
    )
    
    # Vertical flip
    tta_transforms.append(
        transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize']['mean'],
                std=config['normalize']['std']
            )
        ])
    )
    
    # Rotation 90
    tta_transforms.append(
        transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize']['mean'],
                std=config['normalize']['std']
            )
        ])
    )
    
    # Rotation 180
    tta_transforms.append(
        transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['normalize']['mean'],
                std=config['normalize']['std']
            )
        ])
    )
    
    return tta_transforms[:n_augmentations]


class MultiScaleTransform:
    """
    Multi-scale transform for extracting features at different resolutions.
    Useful for microscopy images with details at multiple scales.
    """
    
    def __init__(self, scales=[224, 256, 288], normalize_config=None):
        """
        Initialize multi-scale transform.
        
        Args:
            scales (list): List of image sizes
            normalize_config (dict): Normalization parameters
        """
        self.scales = scales
        if normalize_config is None:
            normalize_config = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        self.normalize_config = normalize_config
        
        self.transforms = []
        for scale in scales:
            self.transforms.append(
                transforms.Compose([
                    transforms.Resize((scale, scale)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=normalize_config['mean'],
                        std=normalize_config['std']
                    )
                ])
            )
    
    def __call__(self, image):
        """
        Apply multi-scale transforms.
        
        Args:
            image: PIL Image
            
        Returns:
            list: List of transformed images at different scales
        """
        return [transform(image) for transform in self.transforms]


class DeiT3Augmentation:
    """
    Simplified 3-Augment recipe from DeiT III (Touvron et al., ECCV 2022).
    
    Uses only 3 augmentations (Grayscale, Solarize, GaussianBlur) with
    RandomResizedCrop and HorizontalFlip. Shown to outperform complex
    augmentation pipelines for fine-grained recognition with ViT backbones.
    
    Args:
        config (dict): Configuration with 'image_size' and 'normalize' keys.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'image_size': 224,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        
        image_size = config.get('image_size', 224)
        norm = config.get('normalize', {})
        mean = norm.get('mean', [0.485, 0.456, 0.406])
        std = norm.get('std', [0.229, 0.224, 0.225])
        
        self.transform = A.Compose([
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.3, 1.0),
                ratio=(0.75, 1.333),
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.ToGray(p=1.0),
                A.Solarize(threshold_range=(0.5, 0.5), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    def __call__(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        augmented = self.transform(image=image)
        return augmented['image']


class ReinhardStainNormalization:
    """
    Reinhard color normalization for microscopy images.
    Normalizes color distribution to match a reference image's statistics.
    Ref: Sevillano et al. 2020 [R6].
    
    Args:
        target_mean: Target mean per channel (L, a, b). Default: empirical microscopy values.
        target_std: Target std per channel (L, a, b). Default: empirical microscopy values.
    """
    
    def __init__(self, target_mean=None, target_std=None):
        self.target_mean = target_mean or [70.0, 0.0, 0.0]
        self.target_std = target_std or [15.0, 5.0, 5.0]
    
    def __call__(self, image):
        """
        Apply Reinhard normalization.
        
        Args:
            image: numpy array (H, W, 3) RGB uint8
            
        Returns:
            Normalized image (H, W, 3) RGB uint8
        """
        import cv2
        
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        for i in range(3):
            channel = lab[:, :, i]
            src_mean = channel.mean()
            src_std = channel.std() + 1e-6
            
            channel = (channel - src_mean) * (self.target_std[i] / src_std) + self.target_mean[i]
            lab[:, :, i] = np.clip(channel, 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return result
    
    @staticmethod
    def compute_dataset_stats(image_paths, max_images=100):
        """
        Compute mean and std of a dataset in LAB space for reference stats.
        
        Args:
            image_paths: List of image paths
            max_images: Max images to sample
            
        Returns:
            (mean_lab, std_lab) each as list of 3 floats
        """
        import cv2
        from PIL import Image
        
        all_means = []
        all_stds = []
        
        sample = image_paths[:max_images]
        for path in sample:
            img = np.array(Image.open(path).convert('RGB'))
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            all_means.append([lab[:, :, i].mean() for i in range(3)])
            all_stds.append([lab[:, :, i].std() for i in range(3)])
        
        mean_lab = np.mean(all_means, axis=0).tolist()
        std_lab = np.mean(all_stds, axis=0).tolist()
        
        return mean_lab, std_lab
