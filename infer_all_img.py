from pathlib import Path
from typing import List

import numpy as np
import cv2
import torch
from tqdm import tqdm

from backbones import get_model


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


def load_model(model_name: str, checkpoint_path: Path) -> torch.nn.Module:
    """
    Load a model from a checkpoint for an inference.

    Args:
        model_name (str): A name of the model.
        checkpoint_path (Path): A path to the checkpoint.

    Returns:
        torch.nn.Module: _description_
    """    
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()
    return net


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
