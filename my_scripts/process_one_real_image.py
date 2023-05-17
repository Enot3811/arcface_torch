"""
Скрипт позволяющий обработать изображение ортофотоплана
(повернуть и обрезать белые края) и сохранить его.
"""


from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.image_tools import read_image, process_raw_real_image


def main(**kwargs):
    img_path: Path = kwargs['source_img_path']
    save_path: Path = kwargs['save_path']
    show_img: bool = kwargs['show']

    img = read_image(img_path)
    processed_img = process_raw_real_image(img)  # Крутим и отрезаем белые края

    if show_img:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(processed_img)
        plt.show()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    

def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Создание датасета из снимка ортофотоплана.'
                     'Подробное описание в файле скрипта.'))

    parser.add_argument(
        'source_img_path', type=Path,
        help='Путь к изображению для обработки.')
    parser.add_argument(
        'save_path', type=Path,
        help='Путь для сохранения обработанного изображения.')
    parser.add_argument(
        '--show', action='store_true',
        help='Отобразить исходный и полученный снимки.')
    args = parser.parse_args()

    if not args.source_img_path.exists():
        raise(FileNotFoundError(
            f'Указанное изображение {args.source_img_path} не существует.'))
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
