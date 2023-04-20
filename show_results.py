from pathlib import Path
import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from inference import inference


def normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def calc_centroids(arr: np.ndarray, keep_deem=False) -> np.ndarray:
    return arr.mean(axis=1, keepdims=keep_deem)


def create_embeddings_plots(embeddings: np.ndarray):
    centroids_embeddings = calc_centroids(embeddings)

    n_columns = 15
    n_rows = np.ceil(
        centroids_embeddings.shape[0] / n_columns).astype(np.int16)

    fig, axs = plt.subplots(n_rows, n_columns)
    fig.set_size_inches(20.0, 20.0, forward=True)
    fig.suptitle('Centroids embeddings.')
    for i in range(centroids_embeddings.shape[0]):
        row = i // n_columns
        column = i % n_columns

        axs[row][column].plot(centroids_embeddings[i])
    return fig


def calc_l2(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.sqrt((arr1 - arr2) ** 2)


def calculate_metrics(embeddings: np.ndarray):
    norm_embeddings = normalize(embeddings)
    
    norm_centroids_embeddings = calc_centroids(norm_embeddings, True)
    
    n_classes, n_samples = norm_embeddings.shape[:2]

    tile_norm_centroids = np.tile(
        norm_centroids_embeddings, (1, n_samples, 1))

    mean_l2 = np.mean(
        calc_l2(tile_norm_centroids, norm_embeddings), axis=(2, 1))

    mean_angles = np.mean(
        calculate_angles(tile_norm_centroids, norm_embeddings), axis=1)

    return mean_l2, mean_angles


def compare_angles_distances(embeddings: np.ndarray):
    # Взять случайный семпл из каждого класса
    # Посчитать расстояние от него до всех центроидов

    # (b, 1, sample_dim)
    centroids = calc_centroids(embeddings, keep_deem=True)
    n_classes, n_samples = embeddings.shape[:2]

    # (b, b)
    angles: List[np.ndarray] = []
    for i in range(n_classes):
        j = random.randint(0, n_samples - 1)
        # embeddings[i, j] (512,)
        # embeddings[i, j][None, :] (1, 512)

        # (b, 1, sample_dim)
        till_sample = np.tile(embeddings[i, j], (n_classes, 1, 1))
        # (b, 1)
        angles.append(calculate_angles(till_sample, centroids))

    angles = np.array(angles).squeeze()
    nearest_idxs = np.argmin(angles, axis=1)
    print('Nearest indexes:', nearest_idxs, sep='\n')
    mask = np.arange(n_classes)
    correct = np.sum(nearest_idxs == mask)
    print('Correct:', correct, 'Total:', n_classes,
          'Accuracy', correct / n_classes)


def main():
    embeddings_path = Path(
        __file__).parents[1] / 'data' / 'img_dataset' / 'results.npy'
    embeddings: np.ndarray = np.load(embeddings_path)

    create_embeddings_plots(embeddings)
    mean_l2, mean_angles = calculate_metrics(embeddings)
    
    print('l2:', mean_l2, sep='\n')
    print('Angles:', mean_angles, sep='\n')

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(mean_l2)
    axs[1].plot(mean_angles)
    # plt.show()

    compare_angles_distances(normalize(embeddings))


    checkpoint_path = 'checkpoints/backbone.pth'
    model_name = 'r50'
    images = ['../data/test_sattelite_112x112.png']

    n_classes = embeddings.shape[0]
    results = [inference(checkpoint_path, model_name, img) for img in images]
    result = normalize(results[0])  # (1, 512)
    angles = calculate_angles(
        calc_centroids(normalize(embeddings), keep_deem=True),
        np.tile(result, (n_classes, 1, 1))).squeeze()
    
    print(np.argmin(angles))


def angular_one2many(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate angles in radians between a vector v1 and a batch of vectors v2.

    Args:
        v1 (np.ndarray): The vector with shape `(vector_dim,)`.
        v2 (np.ndarray): The batch of vectors with shape
        `(n_vectors, vector_dim)`.

    Returns:
        np.ndarray: The angles array with shape `(n_vectors,)`.
    """
    cosine = np.dot(v1, v2.T)/(np.linalg.norm(v1) * np.linalg.norm(v2, axis=1))
    return np.arccos(cosine)


def angular_many2many(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate one to one angles in radians between batches of vectors
    v1 and v2.
    First element of v1 with first element of v2, second with second e.t.c.

    Args:
        v1 (np.ndarray): The batch of vectors with shape
        `(n_vectors, vector_dim)`.
        v2 (np.ndarray): The batch of vectors with shape
        `(n_vectors, vector_dim)`.

    Returns:
        np.ndarray: The angles array with shape `(n_vectors,)`
    """
    cosine = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) *
                                        np.linalg.norm(v2, axis=1))
    return np.arccos(cosine)


if __name__ == '__main__':
    main()
