"""Скрипт для изменения размеров всех изображений в указанной директории."""

from pathlib import Path
import argparse

from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.image_tools import resize_image, read_image, save_image


def main(**kwargs):
    source_dir: Path = kwargs['img_dir']
    save_dir: Path = kwargs['save_dir']
    h, w = kwargs['new_h_w']

    src_img_paths = list(source_dir.glob('*.png'))
    for pth in tqdm(src_img_paths, desc='Обработка изображений'):
        img = resize_image(read_image(pth), (h, w))
        save_image(img, save_dir / pth.name)


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns
    -------
    argparse.Namespace
        Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Изменить размеры изображений в указанной директории, '
                     'сохранить результаты в заданную директорию.'))

    parser.add_argument(
        'img_dir', type=Path,
        help='Путь к директории с исходными изображениями.')
    parser.add_argument(
        'save_dir', type=Path,
        help='Путь к директории для сохранения обработанных изображений.')
    parser.add_argument(
        'new_h_w', type=int, nargs=2,
        help='Размеры для изменения размеров.')
    args = parser.parse_args()

    if not args.img_dir.exists():
        raise FileNotFoundError(
            'Указанная директория с изображениями '
            f'"{args.img_dir}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
