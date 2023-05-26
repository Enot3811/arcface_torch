"""
Скрипт для тестирования модели.

Тестирование проводится на основе сохранённых заранее npy файлов с embeddings.
"""


import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import (normalize, mean_angular_distances, 
                               classify_samples)


def main(**kwargs):
    dset_emb_path = kwargs['dataset_embeddings']
    test_emb_path = kwargs['test_embeddings']

    dset_emb: np.ndarray = np.load(dset_emb_path)  # (n_cls, n_samples, embed)
    test_emb: np.ndarray = np.load(test_emb_path)

    # Если проверяется на датасете, у которого 1 семпл - 1 класс
    if len(test_emb.shape) == 2:
        test_emb = test_emb[:, None, :]

    # Нормализуем
    dset_emb = normalize(dset_emb)
    test_emb = normalize(test_emb)

    # Вычисляем центроиды классов датасета, с которыми будем сравниваться
    dset_centroids = dset_emb.mean(axis=1)
    
    # Средние углы между семплами и их центроидами.
    mean_angles = mean_angular_distances(test_emb, dset_centroids)
    print('Angles:', mean_angles, sep='\n')
    print('Mean angle:', np.mean(mean_angles), sep='\n')

    # Делаем predict, получаем предсказанные классы и среднюю точность
    predicted_cls, cls_accuracy = classify_samples(test_emb, dset_centroids)
    print('Predicted classes', predicted_cls, sep='\n')
    print("Classes' accuracy", cls_accuracy, sep='\n')
    print('Mean accuracy', np.mean(cls_accuracy), sep='\n')


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
        'dataset_embeddings', type=Path,
        help='Путь к файлу с embeddings от датасета.')
    parser.add_argument(
        'test_embeddings', type=Path,
        help='Путь к файлу с тестируемыми embeddings.')
    args = parser.parse_args()

    for path in {args.dataset_embeddings, args.test_embeddings}:
        if not path.exists():
            raise FileNotFoundError(
                f'Указанный файл с embeddings "{path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
