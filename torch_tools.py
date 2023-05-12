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
    Convert an image or a batch of images from tensor to ndarray.

    Args:
        tensor (torch.Tensor): The tensor with shape `[c, h, w]` or
        `[b, c, h, w]`.

    Returns:
        np.ndarray: The array with shape `[h, w, c]` or `[b, h, w, c]`.
    """
    if len(tensor.shape) == 3:
        return tensor.detach().permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 4:
        return tensor.detach().permute(0, 2, 3, 1).numpy()


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert batch of images from ndarray to tensor.

    Args:
        tensor (torch.Tensor): The array with shape `[h, w, c]` or
        `[b, h, w, c]`.

    Returns:
        np.ndarray: The tensor with shape `[c, h, w]` or `[b, c, h, w]`.
    """
    if len(array.shape) == 3:
        return torch.tensor(array.transpose(2, 0, 1))
    elif len(array.shape) == 4:
        return torch.tensor(array.transpose(0, 3, 1, 2))
