import torch
import numpy as np
def show_grid(
    arr: np.ndarray,
    h: int,
    w: int,
    size: Tuple[float, float] = (20.0, 20.0)
):
    fig, axs = plt.subplots(h, w)
    fig.set_size_inches(*size, forward=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(arr.shape[0]):
        row = i // w
        column = i % w
        axs[row][column].get_yaxis().set_visible(False)
        axs[row][column].get_xaxis().set_visible(False)
        axs[row][column].imshow(arr[i])
    return fig, axs


def main():
    # i, j = np.ogrid[:11, :8]
    # arr = 10*i + j
    # print(arr.shape)
    # print(arr, '\n')
    arr = cv2.imread('/home/pc0/projects/arcface/data/road_map.png')
    orig_y, orig_x = arr.shape[:2]
    arr = cv2.resize(arr, dsize=(orig_y // 20, orig_x // 20), interpolation=cv2.INTER_CUBIC)

    w, h, c = arr.shape
    
    x_len = 112
    y_len = 112
    stride = 112

    start = 0
    x_indexer = (
        start + 
        np.expand_dims(np.arange(x_len), 0) +
        np.expand_dims(np.arange(w - x_len - 1, step=stride), 0).T
    )
    y_indexer = (
        start + 
        np.expand_dims(np.arange(y_len), 0) +
        np.expand_dims(np.arange(h - y_len - 1, step=stride), 0).T
    )
    arr = arr[x_indexer][:, :, y_indexer].swapaxes(1, 2)
    h_windows, w_windows = arr.shape[:2]
    arr = arr.reshape(-1, x_len, y_len, c)
    print(arr.shape)

    show_grid(arr, h_windows, w_windows)
    
    augment = get_augmentation()

    for _ in range(3):
        augm_arr = augment(torch.tensor(arr).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        show_grid(augm_arr, h_windows, w_windows)
    
    plt.show()


def get_augmentation():
    return transforms.Compose([
        transforms.RandomPerspective(fill=255, p=1.0),
        transforms.RandomRotation((0, 180), fill=255),
        transforms.RandomApply(
            p=1.0,
            transforms=[transforms.ElasticTransform(alpha=100.0)]),
    ])

