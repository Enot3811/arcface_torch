"""Скрипт для предсказания пачки семплов.

Тестирование проводится на основе сохранённых заранее npy файлов с embeddings.
"""


import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

import numpy as np
import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import normalize, angular_many2many
from my_utils.model_tools import load_model, preprocess_model_input
from my_utils.image_tools import read_image


def main(**kwargs):
    model_path = kwargs['model_path']
    model_name = kwargs['model_name']
    dset_emb_path = kwargs['dataset_embeddings']
    image_dir = kwargs['image_dir']
    cls_names_path = kwargs['cls_names']
    cls_dir = kwargs['classes_images']
    n_nearest = kwargs['n_nearest']
    image_ext = kwargs['image_ext']
    
    # Пути к классифицируемым изображениям
    images_paths: List[Path] = list(image_dir.glob('*'))

    # Подготовка embeddings для сравнения
    # (n_cls, n_dset_samples, embed) или (n_cls, 1, embed)
    dset_emb: np.ndarray = np.load(dset_emb_path)
    # Нормализуем
    dset_emb = normalize(dset_emb)
    # Вычисляем центроиды классов датасета, с которыми будем сравниваться
    dset_centroids = dset_emb.mean(axis=1)  # (n_cls, embed)
    n_cls = dset_centroids.shape[0]

    # Подготовка имён классов
    if cls_names_path is not None:
        with open(cls_names_path) as f:
            idx_to_name = {}
            for line in f.readlines():
                idx, name = line.split()
                idx_to_name[int(idx)] = name
        if len(idx_to_name) != n_cls:
            raise ValueError(
                'Несовпадения количества имён и количества классов. '
                f'{n_cls=}, {len(idx_to_name)=}.')

    # Подготовка изображения классов для отображения предсказаний
    if cls_dir is not None:
        cls_img_paths: List[Path] = list(cls_dir.glob('*'))
        cls_img_paths = filter(lambda path: not path.is_dir(), cls_img_paths)
        # Отсортировать изображения по номерам классов
        cls_img_paths = list(
            sorted(cls_img_paths, key=lambda path: int(path.name[0:-4])))

    # Классификация
    with torch.no_grad():
        model = load_model(model_name, model_path)

        for pth in images_paths:
            sample_img = read_image(pth)
            model_input = preprocess_model_input(sample_img)
            output: torch.Tensor = model(model_input)  # (1, embed_dim)
            img_embeddings = normalize(output.numpy())

            # (1, n_cls)
            distances = angular_many2many(img_embeddings, dset_centroids)[0]
            nearest = np.argsort(distances)[0:n_nearest]
            
            print(f'Изображение {pth.name}:')
            print('Индексы ближайших классов:', nearest)
            if cls_names_path is not None:
                names = [idx_to_name[i] for i in nearest]
                print('Ближайшие классы:', names)
            else:
                names = [str(i) for i in nearest]
            if cls_dir is not None:
                fig, axes = plt.subplots(1, n_nearest + 1, figsize=(16, 16))
                axes[0].imshow(sample_img)
                for j, name in enumerate(names):
                    cls_img = read_image(cls_dir / (name + '.' + image_ext))
                    axes[j + 1].imshow(cls_img)
                plt.show()

            


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Произвести классификацию указанных embeddings.'))

    parser.add_argument(
        'model_path', type=Path,
        help='Путь к весам обученной модели.')
    parser.add_argument(
        'model_name', type=str,
        help='Название модели.')
    parser.add_argument(
        'dataset_embeddings', type=Path,
        help='Путь к файлу с embeddings для сравнения.')
    parser.add_argument(
        'image_dir', type=Path,
        help='Путь к директории с изображениями для классификации.')
    parser.add_argument(
        '--classes_images', type=Path, default=None,
        help=('Путь к директории с классовыми изображениями для отображения '
              'при классификации. Если не задана, то изображения ближайших '
              'классов не будут отображаться'))
    parser.add_argument(
        '--cls_names', type=Path, default=None,
        help=('Путь к файлу с соотношениями индексов и названий классов. '
              'Если не задан, то индексы классов будут считаться '
              'их названиями.'))
    parser.add_argument(
        '--n_nearest', type=int, default=1,
        help=('Количество ближайших классов при классификации, '
              'которые будут отображены. По умолчанию равно 1, то есть '
              'отображаться будет только непосредственно сам predict.'))
    parser.add_argument(
        '--image_ext', type=str, default='jpg', choices=['jpg', 'png'],
        help='Расширение изображений. По умолчанию jpg.')
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f'Указанный файл с весами "{args.model_path}" не существует.')
    if not args.dataset_embeddings.exists():
        raise FileNotFoundError(
            'Указанный файл с embeddings для сравнения '
            f'"{args.dataset_embeddings}" не существует.')
    if args.cls_names is not None and not args.cls_names.exists():
        raise FileNotFoundError(
            'Указанный файл с именами классов '
            f'"{args.cls_names}" не существует.')
    if args.classes_images is not None and not args.classes_images.exists():
        raise FileNotFoundError(
            'Указанная директория с классовыми изображениями '
            f'"{args.classes_images}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
