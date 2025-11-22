import os
import numpy as np


def find_file(folder_path: str, extension: str):
    """Находит первый файл с заданным расширением в директории."""
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            return os.path.join(folder_path, file)
    return None


def find_raw_dirs(root_folder_path: str):
    """Поиск всех папок, содержащих RAW-данные с камеры"""
    raw_dirs = []
    for root, dirs, _ in os.walk(root_folder_path):
        for dir_name in dirs:
            if "raw" in dir_name.lower():
                raw_dirs.append(os.path.join(root, dir_name))
    return sorted(raw_dirs)


def read_cih(folder_path: str):
    """парсинг настроек камеры из CIH хедера"""
    cih_path = find_file(folder_path, ".cih")
    w = h = None
    with open(cih_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match (split := line.split(":"))[0].strip():
                case "Image Width":
                    w = int(split[-1])
                case "Image Height":
                    h = int(split[-1])
                case "Shutter Speed(s)":
                    t = 1 / float(split[-1].split("/")[-1])

            if w and h and t:
                break
    if not (w and h):
        raise ValueError(f"CIH parse failed: {cih_path}")
    return h, w, t


def load_raw_image(path: str, height: int, width: int):
    """формирует nparray из фотографии в указанном пути"""
    return (
        np.fromfile(path, dtype=np.uint16).reshape(height, width, 3).astype(np.float64)
    )
