"""Скрипт для создания датасета из ряда снимков с летающего аппарата.

Каждый снимок из переданной директории обозначается как класс.
Для каждого такого снимка производится аугментация `num_samples` раз
и её результаты сохраняются как семплы в директорию класса.

Датасет сохраняется в формате:
/data_cronsh
├── flight_samples{num_samples}_input{net_input}px
|   ├── train_images
|   |   ├── s0
|   |   |   ├── 0.jpg
|   |   |   ├── 1.jpg
|   |   |   ├── ...
|   |   |   ├── {num_samples - 1}.jpg
|   |   ├── s1
|   |   |   ├── 0.jpg
|   |   |   ├── 1.jpg
|   |   |   ├── ...
|   |   |   ├── {num_samples - 1}.jpg
|   |   ├── ...
|   |   ├── sn
|   |       ├── 0.jpg
|   |       ├── 1.jpg
|   |       ├── ...
|   |       ├── {num_samples - 1}.jpg
|   ├── test_images
|   |   ├── ...
"""


from pathlib import Path
import argparse
from typing import List

from tqdm import tqdm
import torch
import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.image_tools import read_image, resize_image, save_image
from my_utils.torch_tools import (get_augmentation, tensor_to_numpy,
                                  numpy_to_tensor)


def main(**kwargs):
    source_dir: Path = kwargs['source_dir_path']
    train_samples: int = kwargs['num_train']
    test_samples: int = kwargs['num_test']
    net_input_size: int = kwargs['net_input']
    dataset_path: Path = kwargs['save_dir']
    b_size: int = kwargs['batch_size']
    device: str = kwargs['device']

    # Создаём директорию для сохранения датасета
    dataset_path.mkdir(exist_ok=True, parents=True)

    src_img_paths: List[Path] = list(source_dir.glob('*.png'))
    # Отсортировать изображения в порядке возрастания классов
    src_img_paths = list(sorted(src_img_paths, key=lambda path: int(str(path.name)[0:-4])))
    print('Количество классов:', {len(src_img_paths)})

    # Читаем все изображения
    # TODO сделать пакетным образом
    # считать изображения для b_size классов и сделать для них num_samples
    images = []
    desc = 'Чтение и подготовка изображений к генерации'
    for pth in tqdm(src_img_paths, desc=desc):
        images.append(
            resize_image(read_image(pth), (net_input_size, net_input_size)))
    images = np.stack(images)
    

    with torch.no_grad():
        if device == 'auto':
            device: torch.device = (
                torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))
        else:
            device: torch.device = torch.device(device)
        print(f'Using {device} for dataset creating.')

        augmentations = get_augmentation()

        # Генерируем выборки
        for num_samples, directory in zip((train_samples, test_samples),
                                          ('train_images', 'test_images')):
            
            dataset_images_path: Path = dataset_path / directory
            
            # Создаём директории под классы
            cls_dirs: List[Path] = []
            for img_pth in src_img_paths:
                cls_idx = img_pth.name[0:-4]
                dir_path = dataset_images_path / f's{cls_idx}'
                dir_path.mkdir(parents=True, exist_ok=True)
                cls_dirs.append(dir_path)

            # Делаем случайные аугментации и сохраняем их
            set_name = directory.split('_')[0]
            desc = f'Создание {set_name} выборки'
            for i in tqdm(range(num_samples), desc=desc):
                for idx in range(0, len(src_img_paths), b_size):

                    # Набираем батч картинок
                    img_batch = images[idx:idx + b_size]
                    img_batch = numpy_to_tensor(img_batch).to(device=device)

                    augmented_img_batch = augmentations(img_batch).cpu()
                    augmented_img_batch = tensor_to_numpy(augmented_img_batch)

                    for j in range(idx, idx + augmented_img_batch.shape[0]):
                        path = cls_dirs[j] / f'{i}.jpg'
                        save_image(augmented_img_batch[j % b_size], path)


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns
    -------
    argparse.Namespace
        Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Создать датасет из снимков в указанной директории, '
                     'где каждый снимок будет представлять отдельный класс.'))

    parser.add_argument(
        'source_dir_path', type=Path,
        help='Путь к директории с исходными снимками.')
    parser.add_argument(
        'save_dir', type=Path, default=None,
        help=('Директория для сохранения датасета.'))
    parser.add_argument(
        '--num_train', type=int, default=50,
        help=('Количество производимых семплов на класс '
              'для тренировочной выборки.'))
    parser.add_argument(
        '--num_test', type=int, default=50,
        help=('Количество производимых семплов на класс '
              'для тестовой выборки.'))
    parser.add_argument(
        '--net_input', type=int, default=112,
        help=('Размер входа сети, к которому будут приводиться '
              'нарезаемые окна.'))
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Размер батча при создании датасета.')
    parser.add_argument(
        '--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
        help=('Устройство, на котором проводить вычисления. '
              'auto выбирает cuda по возможности.'))
    args = parser.parse_args()

    if not args.source_dir_path.exists():
        raise FileNotFoundError(
            'Указанная директория с изображениями '
            f'"{args.source_dir_path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
