"""Скрипт для тестирования модели.

Тестирование проводится на основе сохранённых заранее npy файлов с embeddings.
"""


import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.np_tools import (normalize, classify_samples, angular_many2many,
                               calculate_accuracy)


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
    n_cls, n_samples, embed_dim = test_emb.shape
    # Выпрямляем в (n_cls * n_samples, embed_dim) для классификации
    angles = angular_many2many(test_emb.reshape(-1, embed_dim), dset_centroids)
    angles = angles.reshape(n_cls, n_samples, n_cls)  # Разворачиваем обратно
    mean_angles = np.mean(angles, axis=1)
    print('Angles:', mean_angles, sep='\n')
    print('Mean angle:', np.mean(mean_angles), sep='\n')

    # Делаем predict, получаем предсказанные классы и среднюю точность
    predict = classify_samples(test_emb.reshape(-1, embed_dim), dset_centroids)
    predict = predict.reshape(n_cls, n_samples)
    ground_truth = np.arange(n_cls)
    ground_truth = np.tile(ground_truth, (n_samples, 1)).T
    cls_accuracy = np.stack([calculate_accuracy(predict[i], ground_truth[i])
                             for i in range(n_cls)])

    print('Predicted classes', predict, sep='\n')
    print("Classes' accuracy", cls_accuracy, sep='\n')
    print('Mean accuracy', np.mean(cls_accuracy), sep='\n')


def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.Namespace: Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Протестировать модель. Посчитать метрики на основе '
                     'указанных embeddings.'))

    parser.add_argument(
        'dataset_embeddings', type=Path,
        help='Путь к файлу с embeddings от датасета.')
    parser.add_argument(
        'test_embeddings', type=Path,
        help='Путь к файлу с тестируемыми embeddings.')
    args = parser.parse_args([
        '/home/pc0/projects/arcface/data/real_images_dataset/win500m_overlap500m_samples500_input224px/embeddings/dataset_embeddings.npy',
        '/home/pc0/projects/arcface/data/real_images_dataset/win500m_overlap500m_samples500_input224px/embeddings/test_embeddings.npy'
    ])

    for path in {args.dataset_embeddings, args.test_embeddings}:
        if not path.exists():
            raise FileNotFoundError(
                f'Указанный файл с embeddings "{path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
