"""Скрипт для создания таблицы соотношения индекса класса с его названием."""


import argparse
from pathlib import Path
from typing import List


def main(**kwargs):
    dataset_dir: Path = kwargs['dataset_dir']
    save_path: Path = kwargs['save_path']

    cls_names = list(map(lambda pth: pth.name, dataset_dir.glob('*')))
    cls_names = list(sorted(cls_names, key=lambda name: int(name[1:])))

    cls_table = [f'{cls_idx} {cls_name}\n'
                for cls_idx, cls_name in enumerate(cls_names)]
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.writelines(cls_table)


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Протестировать модель. '
                     'Посчитать метрики на основе выходных данных модели.'))

    parser.add_argument(
        'dataset_dir', type=Path,
        help=('Путь директории, названия элементов которой будут взяты как '
              'названия классов.'))
    parser.add_argument(
        'save_path', type=Path,
        help='Путь для сохранения таблицы.')
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(('Указанная директория с датасетом '
                                 f'"{args.dataset_dir}" не существует.'))
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
