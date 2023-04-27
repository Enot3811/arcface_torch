from pathlib import Path
from typing import List

import numpy as np
import cv2
import torch
from tqdm import tqdm

from model_tools import load_model
from image_tools import load_images


def main(
    dset_path: Path = Path(
        __file__).parents[1] / 'data' / 'road_dataset_large_images_test',
    model_name: str = 'r50',
    model_path: Path = Path(
        __file__).parent / 'work_dirs' / 'r50_my_conf copy' / 'model.pt'
):    
    """
    Load a model and infer it over images in a defined dir.

    Args:
        dset_path (Path, optional): A path to the dataset of images.
        model_name (str, optional): A name of the model.
        model_path (Path, optional): A path to the model's weights.
    """    
    model = load_model(model_name, model_path)

    directories: List[Path] = list(dset_path.glob('*'))
    directories = list(filter(lambda path: path.is_dir(), directories))
    all_results: List[List[np.ndarray]] = []
    for dir in tqdm(directories):
        images = list(dir.glob('*'))

        images = load_images(images)
        all_results.append(model(images).cpu().detach().numpy())
    
    results = np.array(all_results)
    np.save(dset_path / 'results.npy', results)


def infer_img(img: Path, model: torch.nn.Module) -> np.ndarray:
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(str(img))
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return model(img).cpu().detach().numpy()


if __name__ == '__main__':
    main()
