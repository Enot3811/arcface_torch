from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from inference import inference
from tools import get_sliding_windows


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes,
    # omitting 'exclude' keys, using da values
    intersect = {
        k: v
        for k, v in da.items()
        if (k in db and not any(x in k for x in exclude)
            and v.shape == db[k].shape)
    }
    print(len(da), len(db), len(intersect))
    return intersect


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


def create_augmented_images():
    arr = cv2.imread('/home/pc0/projects/arcface/data/road_map.png')
    orig_y, orig_x = arr.shape[:2]
    arr = cv2.resize(arr, dsize=(orig_y // 20, orig_x // 20), interpolation=cv2.INTER_CUBIC)

    w, h, c = arr.shape
    
    x_len = 112
    y_len = 112
    stride = 112

    arr = get_sliding_windows(arr, y_len, x_len, stride)
    print(arr.shape)

    h_windows, w_windows = arr.shape[:2]

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


def augm1():
    import torchvision
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5
        ),
        torchvision.transforms.RandomRotation(360)
    ])

    path = Path(__file__).parents[1] / 'data' / 'satellite_small'

    dset = torchvision.datasets.ImageFolder(str(path), transforms)
    for i, (image, label) in enumerate(dset):
        cv2.imshow(f'Img {i + 1}', np.array(image))
        key = cv2.waitKey(20000)
        if key == 27:
            break


class RegionGetting:

    def __init__(
        self,
        image_size: Tuple[int, int],
        region_size: Tuple[int, int],
        stride: int = None,
        regions_per_image: int = 2,
        region_margin: int = 1
    ):
        self.region_margin = region_margin
        self.regions_per_image = regions_per_image
        self.image_size = image_size

        h, w = image_size
        w_reg, h_reg = region_size

        if stride is None:
            stride = w_reg
        
        self.x_indexer = (
            torch.arange(0, w_reg)[None, :] +
            torch.arange(0, w - w_reg - 1, stride)[None, :].T
        )
        self.y_indexer = (
            torch.arange(0, h_reg)[None, :] +
            torch.arange(0, h - h_reg - 1, stride)[None, :].T
        )

        self.x_windows = self.x_indexer.size(0)
        self.y_windows = self.y_indexer.size(0)

    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        h, w = self.image_size
        
        available_regions = torch.ones(
            self.x_windows, self.y_windows, dtype=torch.bool)
        
        gotten_regions = []
        while len(gotten_regions) != self.regions_per_image:
            x_idx = torch.randint(0, self.x_windows, ())
            y_idx = torch.randint(0, self.y_windows, ())

            if available_regions[x_idx, y_idx]:
                x_idx = self.x_indexer[x_idx]
                y_idx = self.y_indexer[y_idx]

                gotten_regions.append(img[:, x_idx][:, :, y_idx])
                # TODO Вывести полученный кусочек. Убедиться, что всё ок
                # И вообще продебажить всё, что снизу

                _start_bl = max(0, x_idx)
                _end_bl = min(_start_bl + self.region_margin, w)
                x_blocking = slice(_start_bl, _end_bl)
                _start_bl = max(0, y_idx)
                _end_bl = min(_start_bl + self.region_margin, h)
                y_blocking = slice(_start_bl, _end_bl)
                available_regions[x_blocking, y_blocking] = False
        return gotten_regions


class RegionsDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_directory: Path,
        processing: Callable = None,
        **kwargs
    ):
        self.images = list(map(str, image_directory.rglob('*.jpg')))
        self.processing = processing
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img).permute(2, 0, 1)

        if self.processing is not None:
            img = self.processing(img)
        return img


def main():
    path = Path(__file__).parents[1] / 'data' / 'satellite_small' / 'train'
    dset = RegionsDataset(path)
    img = next(iter(dset))

    img_size = img.shape[1:]
    reg_size = (112, 112)

    reg_get = RegionGetting(img_size, reg_size)
    regions = reg_get(img)


if __name__ == '__main__':
    main()
