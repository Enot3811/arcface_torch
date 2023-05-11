import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_augmentation():
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=(0.5, 1.3), contrast=(0.7, 1.2),
            saturation=(0.7, 1.2), hue=0.1),
        transforms.RandomApply(
            transforms=[transforms.ElasticTransform(alpha=10.0)], p=0.1),
        transforms.RandomPerspective(p=0.8),
        transforms.RandomApply(
            transforms=[transforms.RandomRotation((0, 180))], p=0.8)
    ])


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert batch of images from tensor to ndarray.

    Args:
        tensor (torch.Tensor): The tensor with shape `[b, c, h, w]`.

    Returns:
        np.ndarray: The array with shape `[b, h, w, c]`.
    """    
    return tensor.detach().permute(0, 2, 3, 1).numpy()


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert batch of images from ndarray to tensor.

    Args:
        tensor (torch.Tensor): The array with shape `[b, h, w, c]`.

    Returns:
        np.ndarray: The tensor with shape `[b, c, h, w]`.
    """    
    return torch.tensor(array.transpose(0, 3, 1, 2))
