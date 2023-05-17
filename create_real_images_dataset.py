from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from image_tools import (
    read_image, resize_image, process_raw_real_image,
    get_sliding_windows, get_scaled_shape, show_grid)
from torch_tools import get_augmentation, tensor_to_numpy, numpy_to_tensor


def main(**kwargs):
    img_path = kwargs['source_img_path']
    fov = kwargs['fov']
    orig_scale = kwargs['scale']
    overlap = kwargs['overlap']
    show_cut = kwargs['show_grid']
    save_dset = kwargs['save_dataset']
    num_samples = kwargs['num_samples']
    net_input_size = kwargs['net_input']
    dataset_path = kwargs['save_dir']
    raw = kwargs['raw_source']

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

    augmentations = get_augmentation(color_jitter=True, elastic=True)

    # Отобразить порезанные окна
    # Чтобы корректно работало, необходимо резать без перекрытия
    if show_cut:
        plt.imshow(img)

        n_h_win = h // new_fov
        n_w_win = w // new_fov
        for _ in range(3):
            augmented_windows = augmentations(windows)
            augmented_windows = tensor_to_numpy(augmented_windows)
            show_grid(augmented_windows, n_h_win, n_w_win)
            plt.show()


    # Создаём директории под классы
    if save:
        for i in range(windows.shape[0]):
            dir_path = new_dataset_path / f's{i}'
            dir_path.mkdir(parents=True, exist_ok=True)

        # Делаем случайные аугментации и сохраняем их
        b_size = 1225
        for i in tqdm(range(num_samples)):
            for j in range(0, windows.shape[0], b_size):

                augmented_windows = augmentations(windows[j:j + b_size])
                augmented_windows = tensor_to_numpy(augmented_windows)

                for k in range(j, j + b_size):
                    path = new_dataset_path / f's{k}' / f'{i}.jpg'
                    
                    cv2.imwrite(str(path), cv2.cvtColor(
                        augmented_windows[k % b_size], cv2.COLOR_RGB2BGR))



def parse_args() -> argparse.Namespace:
    """
    Отпарсить передаваемые аргументы.

    Returns:
        argparse.ArgumentParser: Полученные аргументы.
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
        '--save_dataset', action='store_true',
        help='Сохранить датасет.')
    parser.add_argument(
        '--raw_source', action='store_true',
        help=('Провести предварительную предобработку переданного изображения '
              '(повернуть и обрезать белые края).'))
    parser.add_argument(
        '--num_samples', type=int, default=100,
        help='Количество производимых семплов на класс.')
    parser.add_argument(
        '--net_input', type=int, default=112,
        help='Размер входа сети, к которому будут приводиться нарезаемые окна.'
    )
    parser.add_argument(
        '--save_dir', type=Path, default=None,
        help=('Директория для сохранения датасета. '
              'Если не указана, то указывается директория в '
              '"project_dir/../data/real_images_dataset/" с названием, '
              'созданным из переданных аргументов.'))
    args = parser.parse_args()

    # Если шаг 0, то прировнять его к fov. То есть нарезка без перекрытия.
    if args.overlap == 0:
        args.overlap = args.fov
    # Если необходимо, сгенерировать название для папки датасета
    if args.save_dir is None and args.save_dataset:
        args.save_dir = (
            Path(__file__).parents[1] / 'data' / 'real_images_dataset' /
            (f'win{args.fov}m_overlap{args.overlap}m_'
             f'samples{args.num_samples}_input{args.net_input}px'))
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
