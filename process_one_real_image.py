from pathlib import Path
from image_tools import read_image, process_raw_real_image
import cv2
import matplotlib.pyplot as plt


img_path = (Path(__file__).parents[1] /
            'data' / 'real_data_raw' / 'B075.jpg')
img = read_image(img_path)
img = process_raw_real_image(img)  # Крутим и отрезаем белые края
# plt.imshow(img)
# plt.show()
cv2.imwrite(
    str(img_path.parents[1] / 'real_images_dataset' / 'B075_processed.jpg'),
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
