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
    n_columns = 15
    n_rows = np.ceil(
        embeddings.shape[0] / n_columns).astype(np.int16)

    fig, axs = plt.subplots(n_rows, n_columns)
    fig.set_size_inches(20.0, 20.0, forward=True)
    fig.suptitle('Centroids embeddings.')
    for i in range(embeddings.shape[0]):
        row = i // n_columns
        column = i % n_columns

        axs[row][column].plot(embeddings[i])
    return fig


def calc_l2(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.sqrt((arr1 - arr2) ** 2)


def mean_angular_distances(
    embeddings: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """
    Берётся центроид и ембединги семплов каждого класса и вычисляется
    средний угол между ембедингами и центроидом соответствующего класса.

    Args:
        embeddings (np.ndarray): shape is `(n_classes, n_samples, embed_dim)`.
        centroids (np.ndarray): shape is `(n_classes, embed_dim)`.

    Returns:
        np.ndarray: shape is `(n_classes,)`
    """
    n_classes = embeddings.shape[0]
    classes_distances = []
    for cls in range(n_classes):
        cls_centroid = centroids[cls]
        cls_embeddings = embeddings[cls]
        classes_distances.append(
            angular_one2many(cls_centroid, cls_embeddings))

    classes_distances = np.array(classes_distances)
    mean_classes_distances = np.mean(classes_distances, axis=1)
    return mean_classes_distances
    

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


def angular_to_centroids(embeddings: np.ndarray, centroids: np.ndarray):
    # Берётся случайный семпл из каждого класса и вычисляются расстояния
    # до всех центроидов
    # shape (n_classes, n_classes)
    n_classes, n_samples = embeddings.shape[:2]

    classes_distances = []
    for i in range(n_classes):
        j = random.randint(0, n_samples - 1)
        rand_sample = embeddings[i, j]

        classes_distances.append(
            angular_one2many(rand_sample, centroids))
    classes_distances = np.array(classes_distances)

    nearest_idxs = np.argmin(classes_distances, axis=1)
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
