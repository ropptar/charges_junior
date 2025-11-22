from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import config
from modules import files, imagetools, plottools


def process_raw_folder(raw_folder_path: str, count: int):
    """
    Обработка RAW-данных в указанной папке и сохранение кадров и построенных графиков в {config.OUTPUT_DIR}/{count}

    Args:
        raw_folder_path: Путь до папки
        count: Номер выходной папки
    """
    # Создание папки для результатов
    output_folder = config.OUTPUT_DIR / str(count)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Чтение CIH
    try:
        heigth, width, exposure = files.read_cih(raw_folder_path)
    except Exception as e:
        print(f"Error reading CIH in {raw_folder_path}: {str(e)}")
        return

    # Обработка RAW-файлов
    raw_paths = np.array(sorted(Path(raw_folder_path).glob("*.raw")))
    raw_files = np.array(
        [files.load_raw_image(raw_path, height, width) for raw_path in raw_paths]
    )
    indices = np.array(DARK_CONFIG[count])

    mask = np.ones(len(raw_files), dtype=bool)
    mask[indices] = False
    dark_img = np.mean(raw_files[indices], axis=0)
    charge_images = raw_files[mask]

    for idx, img in enumerate(charge_images):
        subtr_img = img - dark_img
        subtr_img = np.array(subtr_img)
        subtr_img = np.clip(subtr_img, 0, None)
        corrected = imagetools.shear_image_y(subtr_img, config.SHEAR_MAGNITUDE[count])

        # Обрезка
        y_start = int(height * config.CROP_CONFIG[count]["y_start_ratio"])
        y_end = int(height * config.CROP_CONFIG[count]["y_end_ratio"])
        cropped = corrected[y_start:y_end, :]
        cropped = np.clip(cropped, 0, None)
        fig, axes = plt.subplots(2, 1, figsize=(8, 3))
        axes[0].imshow(img / np.max(img))
        axes[1].imshow(cropped / np.max(cropped))
        plt.savefig(output_folder / f"frame_{idx:02d}.png")
        plt.close()
        # Сохранение

        fig = plottools.add_wavelength_scale(
            img,
        )
        fig.savefig(
            output_folder / f"origframe_{idx:02d}.png", bbox_inches="tight", dpi=200
        )
        plt.close(fig)


def start():
    print("Input dir is:", config.INPUT_DIR)
    print("Output dir is:", config.OUTPUT_DIR)
    if input("Continue? y/n: ")[0].lower() != "y":
        return

    if not (raw_folders := files.find_raw_dirs(config.INPUT_DIR)):
        print("No raw folders found!")
        return

    # todo process loop
    for count, folder in enumerate(raw_folders):
        print("\nProcessing folder:", folder)
        process_raw_folder(folder, count + 1)

    print("\nAll done! Results saved to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    start()
