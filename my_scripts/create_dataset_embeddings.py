"""Скрипт для создания embeddings на основе указанного датасета.
"""


import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.model_tools import load_model, preprocess_model_input
from my_utils.image_tools import read_image


def main(**kwargs):
    dset_path: Path = kwargs['dataset_path']
    save_path: Path = kwargs['save_path']
    model_path: Path = kwargs['model_path']
    model_name: str = kwargs['model_name']
    b_size: int = kwargs['batch_size']

    cls_dirs: List[Path] = list(dset_path.glob('*'))
    # Отфильтровать только директории на случай,
    # если внутри будет что-то лишнее
    cls_dirs = filter(lambda path: path.is_dir(), cls_dirs)
    # Отсортировать директории в порядке номеров классов
    cls_dirs = list(sorted(cls_dirs, key=lambda path: int(str(path.name)[1:])))

    with torch.no_grad():
        device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

        model = load_model(model_name, model_path).to(device=device)

        dset_embeddings: List[List[np.ndarray]] = []
        desc = 'Обработка датасета'
        # Перебираем директории классов
        for dir in tqdm(cls_dirs, desc=desc):
            cls_images_paths = list(dir.glob('*'))

            # Перебираем изображения класса
            cls_embeddings = []
            for i in range(0, len(cls_images_paths), b_size):
                batch_paths = cls_images_paths[i: i + b_size]
                # Набираем батч картинок
                img_batch = []
                for img_path in batch_paths:
                    img = read_image(img_path)
                    img = preprocess_model_input(img)
                    img_batch.append(img)
                # Отправляем в сеть, сохраняем выход
                img_batch = torch.cat(img_batch, axis=0).to(device=device)
                embeddings = model(img_batch).cpu().numpy()
                cls_embeddings.append(embeddings)

            # Embeddings класса добавляем ко всем
            cls_embeddings = np.concatenate(cls_embeddings, axis=0)
            dset_embeddings.append(cls_embeddings)
        
    dset_embeddings = np.stack(dset_embeddings, axis=0)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, dset_embeddings)


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns
    -------
    argparse.Namespace
        Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Создать embeddings из датасета.'))

    parser.add_argument(
        'dataset_path', type=Path,
        help='Путь к директории с датасетом.')
    parser.add_argument(
        'save_path', type=Path,
        help='Путь для сохранения npy файла с embeddings.')
    parser.add_argument(
        'model_path', type=Path,
        help='Путь к весам обученной модели.')
    parser.add_argument(
        'model_name', type=str,
        help='Название модели.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Размер батча, отправляемого в сеть.')
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(
            'Указанная директория с датасетом '
            f'"{args.dataset_path}" не существует.')
    if not args.model_path.exists():
        raise FileNotFoundError(
            f'Файл с весами модели "{args.model_path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
