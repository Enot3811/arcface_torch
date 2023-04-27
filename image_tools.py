from typing import Optional, Union, List
from pathlib import Path

import numpy as np
import cv2
import torch


def read_image(path: Union[Path, str], grayscale: bool = False) -> np.ndarray:
    """
    Read image to numpy array.
    Parameters
    ----------
    path : Union[Path, str]
        Path to image file
    grayscale : bool, optional
        Whether read image in grayscale, by default False
    Returns
    -------
    np.ndarray
        Array containing read image.
    Raises
    ------
    FileNotFoundError
        Did not find image.
    ValueError
        Image reading is not correct.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Did not find image {path}.')
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError('Image reading is not correct.')
    img = cv2.cv2tColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_sliding_windows(
    source_image: np.ndarray,
    h_win: int,
    w_win: int,
    stride: Optional[int] = None
) -> np.ndarray:
    """
    Cut a given image into windows with defined shapes and stride.

    Args:
        source_image (np.ndarray): The original image.
        h_win (int): Height of the windows.
        w_win (int): Width of the windows.
        stride (Optional[int]): The stride of the sliding windows.
        If not defined it will be set by w_win value.

    Returns:
        np.ndarray: The cut image with shape `[num_windows, h_win, w_win, c]`.
    """    
    w, h, c = source_image.shape

    if stride is None:
        stride = w_win

    x_indexer = (
        np.expand_dims(np.arange(w_win), 0) +
        np.expand_dims(np.arange(w - w_win - 1, step=stride), 0).T
    )
    y_indexer = (
        np.expand_dims(np.arange(h_win), 0) +
        np.expand_dims(np.arange(h - h_win - 1, step=stride), 0).T
    )
    windows = source_image[x_indexer][:, :, y_indexer].swapaxes(1, 2)
    windows = windows.reshape(-1, w_win, h_win, c)
    return windows


def load_images(image_paths: List[Path]) -> torch.Tensor:
    """
    Load and prepare images to pass into a network.

    Args:
        image_paths (List[Path]): Paths to the needed images.

    Returns:
        torch.Tensor: Loaded and processed images with shape
        `[num_images, c, h, w].`
    """    
    images = []
    for img in image_paths:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        images.append(img)
    return torch.cat(images)