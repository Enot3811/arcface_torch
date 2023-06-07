import torch
import torchvision.transforms as transforms
import numpy as np


def get_augmentation(
    color_jitter: bool = True,
    blur: bool = True,
    random_crop: bool = True,
    random_perspective: bool = True,
    random_rotate: bool = True
) -> transforms.Compose:
    transf = []
    if color_jitter:
        transf.append(
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.5, 1.5),
                                   saturation=(0.7, 1.3), hue=0.05))
    if blur:
        transf.append(transforms.RandomApply(
            transforms=[transforms.GaussianBlur(kernel_size=3)], p=0.2))
    if random_perspective:
        transf.append(transforms.RandomPerspective(p=0.2))
    if random_rotate:
        transf.append(transforms.RandomApply(
            transforms=[transforms.RandomRotation((0, 180))], p=0.8))
    if random_crop:
        transf.append(transforms.RandomResizedCrop(
            (224, 224), (0.5, 1.0), ratio=(0.7, 1.0), antialias=True))
            
    return transforms.Compose(transf)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert an image or a batch of images from tensor to ndarray.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor with shape `[c, h, w]` or `[b, c, h, w]`.

    Returns
    -------
    np.ndarray
        The array with shape `[h, w, c]` or `[b, h, w, c]`.
    """
    if len(tensor.shape) == 3:
        return tensor.detach().permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 4:
        return tensor.detach().permute(0, 2, 3, 1).numpy()


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert batch of images from ndarray to tensor.

    Parameters
    ----------
    array : np.ndarray
        The array with shape `[h, w, c]` or `[b, h, w, c]`.

    Returns
    -------
    np.ndarray
        The tensor with shape `[c, h, w]` or `[b, c, h, w]`.
    """
    if len(array.shape) == 3:
        return torch.tensor(array.transpose(2, 0, 1))
    elif len(array.shape) == 4:
        return torch.tensor(array.transpose(0, 3, 1, 2))
