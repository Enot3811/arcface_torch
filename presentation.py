from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt

from image_tools import read_image, resize_image, get_scaled_shape
from torch_tools import get_augmentation, numpy_to_tensor, tensor_to_numpy


def show_similarity_heatmap(angles: np.ndarray, percent_of_overlapping):
    """
    Нарисовать тепловую карту похожести.

    Args:
        angles (np.ndarray): Угловое расстояние с размером
        `[h_pieces, w_pieces]`
    """
    # Надо узнать сколько окон чистых
    # Насколько эти чистые окна перекрываются соседями
    # Всех этих соседей либо отбросить, либо примешать
    # Стоит посмотреть результаты как они есть
    pass


def make_windows_grid(
    image: np.ndarray, win_size: int, stride: int,
    color: Optional[Tuple[int, int, int]] = (255, 0, 0),
    thickness: Optional[int] = 1,
    special_point: Optional[Tuple] = None
) -> np.ndarray:
    """
    Нарисовать сетку из перекрывающих окон на изображении.

    Args:
        image (np.ndarray): Исходное изображение.
        win_size (int): Размер окна.
        stride (int): Шаг окна.
        color (Optional[Tuple[int, int, int]], optional): Цвет рисуемых рамок.
        thickness (Optional[int], optional): Толщина рисуемых рамок.

    Returns:
        np.ndarray: Отредактированная картинка.
    """    
    h, w = image.shape[:2]
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            if i % win_size == 0 and j % win_size == 0:
                image = cv2.rectangle(
                    image, (j, i), (j + win_size, i + win_size),
                    color, thickness * 2)
            else:
                image = cv2.rectangle(
                    image, (j, i), (j + win_size, i + win_size),
                    color, thickness)
    if special_point != None:
        i, j = special_point
        image = cv2.rectangle(
            image, (j, i), (j + win_size, i + win_size),
            (0, 255, 0), thickness * 2)
    return image



def main():
    # Здесь делались картинки для презентации

    # Сетка с окнами
    resize_shape = (1400, 900)
    window_size = 100
    stride = 100
    map_path = Path(__file__).parents[1] / 'data' / 'road_map.png'
    map_image = read_image(map_path)
    map_image = resize_image(map_image, (resize_shape[0], resize_shape[1]))
    grid = map_image.copy()
    grid = make_windows_grid(grid, window_size, stride)
    plt.imshow(grid)

    # Пример аугментации кусочка с карты дорог
    piece = map_image[
        4 * window_size:5 * window_size, 7 * window_size : 8 * window_size]
    augmentation = get_augmentation(color_jitter=False, elastic=False)
    augmented_piece = tensor_to_numpy(augmentation(numpy_to_tensor(piece)))
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(piece)
    axs[1].imshow(augmented_piece)

    # Пример аугментации кусочка с настоящей картинки
    fig, ax = plt.subplots(1, 1)
    img_path = (Path(__file__).parents[1] /
                'data' / 'real_images_dataset' / 'B075_processed.jpg')
    img = read_image(img_path)
    h_orig, w_orig = img.shape[:2]
    orig_scale = 0.5  # Метров в пикселе
    net_input_size = 112
    fov = 200  # Сторона квадрата поля зрения в метрах
    overlap = 30  # Шаг перекрывающего окна в метрах
    
    h, w, scaled_overlap_px, new_scale = get_scaled_shape(
        h_orig, w_orig, orig_scale, overlap, fov, net_input_size)
    new_fov = net_input_size

    img = resize_image(img, (h, w))
    ax.imshow(img)


    fig, axs = plt.subplots(1, 2)
    piece = img[20 * new_fov: 21 * new_fov, 23 * new_fov: 24 * new_fov]
    augmentation = get_augmentation(color_jitter=True, elastic=True)
    augmented_piece = tensor_to_numpy(augmentation(numpy_to_tensor(piece)))
    axs[0].imshow(piece)
    axs[1].imshow(augmented_piece)

    # Сетка на реальной карте
    grid = img.copy()
    grid = make_windows_grid(img, new_fov, scaled_overlap_px,
                             special_point=(20 * new_fov, 23 * new_fov))
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grid)

    plt.show()


    # # Берём заранее сохранённые ембединги от кусков карты,
    # # которые будем предсказывать
    # embeddings_path = (Path(__file__).parents[1] / 'data' /
    #                    'road_dataset_small_images' / 'results.npy')
    # embeddings: np.ndarray = np.load(embeddings_path)

    # # Их необходимо нормализовать
    # norm_embed = normalize(embeddings)
    # embed_centroids = norm_embed.mean(axis=1)

    # # Нужна модель и какая-нибудь картинка, которую хотим предсказать
    # checkpoint_path = 'checkpoints/backbone.pth'
    # model_name = 'r50'
    # image = '../data/test_satellite_112x112.png'

    # n_classes = embeddings.shape[0]
    # results = inference(checkpoint_path, model_name, image)
    # result = normalize(results[0]).squeeze()  # (1, 512)
    # angles = angular_one2many(result, embed_centroids)

    # # Сколько кусков в строке и столбце
    # n_rows = 9
    # n_columns = 14
    # angles.reshape(())

    # sns.heatmap(accuracy_per_class, annot=True, ax=ax, annot_kws={'size': 25})
    
    # print('Предсказанный класс:', np.argmin(angles))


if __name__ == '__main__':
    main()
    # Взять настоящую картинку
    # Отрезать от неё кусок, покрутить