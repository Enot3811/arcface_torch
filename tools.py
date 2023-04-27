import numpy as np


def get_sliding_windows(
    source_image: np.ndarray,
    h_win: int,
    w_win: int,
    stride: int
) -> np.ndarray:
    """
    Cut a given image into windows with defined shapes and stride.

    Args:
        source_image (np.ndarray): The original image.
        h_win (int): Height of the windows.
        w_win (int): Width of the windows.
        stride (int): The stride of the sliding windows.

    Returns:
        np.ndarray: The cut image with shape `[num_windows, h_win, w_win, c]`.
    """    
    w, h, c = source_image.shape

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