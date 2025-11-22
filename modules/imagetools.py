import numpy as np
from cv2 import warpAffine, INTER_LANCZOS4


def shear_image_y(image: np.array, shear_magnitude: dict):
    """Вертикальный сдвиг с обнулением краев"""
    h, w = image.shape[:2]
    M = np.array([[1, 0, 0], [shear_magnitude / w, 1, 0]], dtype=np.float32)

    return warpAffine(image, M, (w, h), flags=INTER_LANCZOS4, borderValue=0)
