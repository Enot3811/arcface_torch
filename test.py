from pathlib import Path

import torch
import cv2
import numpy as np

from model_tools import inference
from show_results import normalize, classify_samples, angular_one2many


def main():
    """
    Последнее, что было здесь, это тест последнего обучения на карте дорог
    Data_612000_cls_6120
    Здесь можно посмотреть точность модели и сделать предикт
    какой-нибудь картинки.
    """
    # Parameters
    model_path = (
        '/home/pc0/projects/arcface/arcface_torch/work_dirs/' +
        'r50_my_conf/model.pt')
    embeddings_path = Path(
        '/home/pc0/projects/arcface/data/' +
        'road_dataset_large_images_test/results.npy')
    map_embeddings_path = Path(
        '/home/pc0/projects/arcface/data/cut_map/results.npy')

    # model = load_model('r50', model_path)

    embeddings: np.ndarray = np.load(embeddings_path)
    centroids: np.ndarray = np.load(map_embeddings_path)
    centroids = centroids.squeeze()

    norm_embed = normalize(embeddings)
    embed_centroids = normalize(centroids)

    _, accuracy_per_class = classify_samples(norm_embed, embed_centroids)
    # accuracy_per_class = accuracy_per_class.reshape(n_rows, n_columns)
    print('Accuracy:', np.mean(accuracy_per_class))

    # реальная картинка
    test_img = '/home/pc0/projects/arcface/data/test_satellite_112x112.png'
    result = inference(model_path, 'r50', test_img)
    result = normalize(result.squeeze())  # (1, 512)
    angles = angular_one2many(result, embed_centroids)
    indexes = np.argsort(angles)
    print('Predicted class:', np.argmin(angles))
    print('Nearest:', indexes[:10])


if __name__ == '__main__':
    main()
