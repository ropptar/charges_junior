import numpy as np


def apply_fft_filter(image, count):
    filtered_channels = []
    for i in range(3):  # Обработка каждого цветового канала
        # FFT с центрированием нулевой частоты
        fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))

        s = np.log(np.abs(fft) + 1e-8)
        s /= np.max(s)
        # Image.fromarray( (s*255).astype(np.uint8)).save(f"FFT_{count}_{i}.png")
        '''plt.figure(figsize=(14,4))
        plt.imshow(np.log(np.abs(fft) + 1e-8), cmap='gray')
        plt.title(f"Частотная область (лог амплитуда) {count}_{i}")
        plt.tight_layout()
        plt.savefig(f"FFT_{count}_{i}.png", dpi = 300)
        plt.close()'''
        # Пример маски: обнуление вертикальной линии (настройка под шум)
        mask = np.ones_like(np.abs(fft))
        center_y, center_x = fft.shape[0] // 2, fft.shape[1] // 2
        # print(center_x)
        mask[:, center_x-1:center_x + 1] = 0  # Вертикальная полоса

        # Обратное преобразование
        filtered_fft = fft * mask
        filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
        filtered_channels.append(np.abs(filtered_img))

    # Сборка обратно в RGB
    return np.stack(filtered_channels, axis=-1)


def subtract_dark(from_img, dark_img):
    """Вычитает темновой кадр из изображения"""
    return from_img - dark_img