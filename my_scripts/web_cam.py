import cv2
from pathlib import Path
import shutil

import sys
sys.path.append(str(Path(__file__).parents[1]))
from my_utils.image_tools import resize_image


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 24) # Частота кадров
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Ширина кадров в видеопотоке.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960) # Высота кадров в видеопотоке.

    out_dir = Path(__file__).parents[2] / 'camera_out'
    # Если ранее уже существовала такая директория, то удаляем её
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # и создаём новую пустую
    out_dir.mkdir(exist_ok=True, parents=True)

    count = 0
    while True:
        ret, img = cap.read()
        print(ret)
        cy = img.shape[0] // 2
        cx = img.shape[1] // 2
        win = 224  # Вырезаемой зоны из получаемого кадра
        # Вырезается из середины
        img = img[cy - win // 2:cy + win // 2, cx - win // 2:cx + win // 2]
        if win != 224:  # Если не совпадает с размером сети, то ресайз
            img = resize_image(img, (224, 224))
        
        cv2.imshow('camera', img)
        img_pth = out_dir / (str(count) + '.jpg')
        cv2.imwrite(str(img_pth), img)
        if cv2.waitKey(10) == 27: # Клавиша Esc
            break
        count += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
