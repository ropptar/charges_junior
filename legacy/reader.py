import os
import numpy as np

def find_file(directory, extension):
    """Находит первый файл с заданным расширением в директории."""
    for file in os.listdir(directory):
        if file.lower().endswith(extension.lower()):
            return os.path.join(directory, file)
    return None


def read_cih(directory):
    cih_path = find_file(directory, '.cih')
    w = h = None
    with open(cih_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Image Width' in line: w = int(line.split(':')[-1])
            if 'Image Height' in line: h = int(line.split(':')[-1])
            if 'Shutter Speed(s)' in line: t = 1/float(line.split(':')[-1].split('/')[-1])
            if w and h and t: break
    if not (w and h): raise ValueError(f'CIH parse failed: {cih_path}')
    return h, w, t


def load_raw(path, height, width):
    raw_image = np.fromfile(path, dtype=np.uint16)
    raw_image = raw_image.reshape(height, width, 3)
    return raw_image.astype(np.float64)


def find_raw_dirs(root_path):
    """Находит все директории с 'raw' в названии"""
    raw_dirs = []
    for root, dirs, _ in os.walk(root_path):
        for dir_name in dirs:
            if 'raw' in dir_name.lower():
                raw_dirs.append(os.path.join(root, dir_name))
    return raw_dirs
