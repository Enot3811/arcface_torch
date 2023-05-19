"""Скрипт для создания датасета из снимка ортофотоплана.

Снимок при необходимости предобрабатывается (поворот и обрезка белых краёв),
затем на основе переданных размеров окна, перекрытия, масштаба и
размера входа сети вычисляются новые размеры и масштаб изображения, чтобы
одно окно по размерам соответствовала и указанным размерам в метрах и
входу сети в пикселях.
Полученное масштабированное изображение нарезается скользящим окном.
Каждое из окон после этого начинает представлять определённый класс.
Для каждого класса берётся соответствующее окно и аугментируется указанное
количество раз, производя тем самым семплы для класса.

Датасет сохраняется в формате:
/real_images_dataset
├── win{win_size}m_overlap{overlap}m_samples{num_samples}_input{net_input}px
|   ├── images
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
|   ├── cut_image
|       ├── s0.jpg
|       ├── s1.jpg
|       ├── ...
|       ├── s{num_classes - 1}.jpg
"""


from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.image_tools import (
    read_image, resize_image, process_raw_real_image,
    get_sliding_windows, get_scaled_shape, show_grid, save_image)
from my_utils.torch_tools import get_augmentation, tensor_to_numpy, numpy_to_tensor


def main(**kwargs):
    img_path = kwargs['source_img_path']
    fov = kwargs['fov']
    orig_scale = kwargs['scale']
    overlap = kwargs['overlap']
    show_cut = kwargs['show_grid']
    train_samples = kwargs['num_train']
    test_samples = kwargs['num_test']
    net_input_size = kwargs['net_input']
    dataset_path = kwargs['save_dir']
    raw = kwargs['raw_source']
    b_size = kwargs['batch_size']

    # Если шаг 0, то прировнять его к fov. То есть нарезка без перекрытия.
    if overlap == 0:
        overlap = fov
    # Если необходимо, сгенерировать название для папки датасета
    if dataset_path is None:
        dataset_path = (
            Path(__file__).parents[2] / 'data' / 'real_images_dataset' /
            (f'win{fov}m_overlap{overlap}m_'
             f'samples{train_samples}_input{net_input_size}px'))
        print('Директория для сохранения датасета не указана. '
              'Датасет будет сохранён в следующей директории:\n'
              f'"{dataset_path}"\n')

    img = read_image(img_path)
    if raw:
        img = process_raw_real_image(img)  # Крутим и отрезаем белые края
    h_orig, w_orig = img.shape[:2]

    print('Исходное изображение:', f'Размеры: {h_orig}, {w_orig}',
          f'Масштаб: {orig_scale} м/px', f'Поле зрения: {fov} м',
          f'Перекрывающий шаг: {overlap} м', sep='\n')
    
    h, w, scaled_overlap_px, new_scale = get_scaled_shape(
        h_orig, w_orig, orig_scale, overlap, fov, net_input_size)
    new_fov = net_input_size
    
    print('\nОбработанное изображение:', f'Размеры: {h}, {w}',
          f'Масштаб: {new_scale} м/px', f'Поле зрения: {new_fov} px',
          f'Перекрывающий шаг: {scaled_overlap_px} px', sep='\n')

    img = resize_image(img, (w, h))

    windows = get_sliding_windows(
        img, new_fov, new_fov, scaled_overlap_px)
    
    print('Порезанные окна:', windows.shape)

    with torch.no_grad():
        device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('cpu'))
        print(f'\nUsing {device} for dataset creating.')
        augmentations = get_augmentation(color_jitter=True, elastic=True)

        # Отобразить порезанные окна
        # Чтобы корректно работало, необходимо резать без перекрытия
        if show_cut:
            n_h_win = h // new_fov
            n_w_win = w // new_fov
            for _ in range(3):
                augmented_windows = augmentations(numpy_to_tensor(windows).
                                                  to(device=device)).cpu()
                augmented_windows = tensor_to_numpy(augmented_windows)
                show_grid(augmented_windows, n_h_win, n_w_win)
                plt.show()

        # Сохраняем порезанные окна без аугментаций
        cut_image_path = dataset_path / 'cut_image'
        cut_image_path.mkdir(parents=True, exist_ok=True)
        desc = 'Сохранение нарезанных окон'
        for i in tqdm(range(windows.shape[0]), desc=desc):
            save_image(windows[i], cut_image_path / f's{i}.jpg')

        # Создаём выборки
        for num_samples, directory in zip((train_samples, test_samples),
                                          ('train_images', 'test_images')):
            dataset_images_path = dataset_path / directory

            # Создаём директории под классы
            for i in range(windows.shape[0]):
                dir_path = dataset_images_path / f's{i}'
                dir_path.mkdir(parents=True, exist_ok=True)

            # Делаем случайные аугментации и сохраняем их
            set_name = directory.split('_')[0]
            desc = f'Создание {set_name} выборки'
            for i in tqdm(range(num_samples), desc=desc):
                for cls_idx in range(0, windows.shape[0], b_size):

                    # Набираем батч окон
                    win_batch = windows[cls_idx: cls_idx + b_size]
                    win_batch = numpy_to_tensor(win_batch).to(device=device)
                    augmented_win_batch = augmentations(win_batch).cpu()
                    augmented_win_batch = tensor_to_numpy(augmented_win_batch)

                    for j in range(cls_idx,
                                   cls_idx + augmented_win_batch.shape[0]):
                        path = dataset_images_path / f's{j}' / f'{i}.jpg'
                        save_image(augmented_win_batch[j % b_size], path)


def parse_args() -> argparse.Namespace:
    """Отпарсить передаваемые аргументы.

    Returns
    -------
    argparse.Namespace
        Полученные аргументы.
    """
    parser = argparse.ArgumentParser(
        description=('Создание датасета из снимка ортофотоплана.'
                     'Подробное описание в файле скрипта.'))

    parser.add_argument(
        'source_img_path', type=Path,
        help='Путь к снимку для нарезки.')
    parser.add_argument(
        'fov', type=int,
        help='Размер стороны нарезаемого окна в метрах.')
    parser.add_argument(
        'scale', type=float,
        help='Масштаб передаваемого снимка в метрах на пиксель.')
    parser.add_argument(
        '--overlap', type=int, default=0,
        help='Размер шага окна или перекрытие в метрах.')
    parser.add_argument(
        '--show_grid', action='store_true',
        help='Отобразить порезанный снимок.')
    parser.add_argument(
        '--raw_source', action='store_true',
        help=('Провести предварительную предобработку переданного изображения '
              '(повернуть и обрезать белые края).'))
    parser.add_argument(
        '--num_train', type=int, default=0,
        help=('Количество производимых семплов на класс '
              'для тренировочной выборки.'))
    parser.add_argument(
        '--num_test', type=int, default=0,
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
        '--save_dir', type=Path, default=None,
        help=('Директория для сохранения датасета. '
              'Если не указана, то указывается директория в '
              '"project_dir/../data/real_images_dataset/" с названием, '
              'созданным из переданных аргументов.'))
    args = parser.parse_args()

    if not args.source_img_path.exists():
        raise FileNotFoundError(
            'Указанное изображение для нарезки '
            f'"{args.source_img_path}" не существует.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
