"""
Интерактивный выбор точек клавишами Z и X с автоматическим замыканием разрывов.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
from collections import deque
import cv2

sys.path.append(os.path.dirname(__file__))

try:
    from train_maze_model import UNet
    from real_map_config import RealMapConfig
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Предупреждение: UNet не загружен. Используем бинаризацию исходного изображения.")

# ---------- Параметры ----------
IMAGE_PATH = "data/real_maps/val/images/sample_00000.png"   # измените
MODEL_PATH = "models_real_maps/real_map_best_iou.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
CLOSE_KERNEL_SIZE = 3   # размер ядра замыкания (3,5,7) – чем больше, тем сильнее склеиваются дороги

# ---------- Функции ----------
def invert_image(x):
    return 1 - x

def load_model(path):
    if not MODEL_AVAILABLE or not os.path.exists(path):
        return None
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_road_mask(model, image_path):
    img = Image.open(image_path).convert('L')
    orig_size = img.size
    transform = T.Compose([
        T.Resize((RealMapConfig.IMG_SIZE, RealMapConfig.IMG_SIZE)),
        T.ToTensor(),
        T.Lambda(invert_image)
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).cpu().squeeze().numpy()
    mask_small = (prob > THRESHOLD).astype(np.uint8)
    mask_img = Image.fromarray((mask_small * 255).astype(np.uint8))
    mask_img = mask_img.resize(orig_size, Image.NEAREST)
    return np.array(mask_img) // 255

def load_road_mask_from_image(image_path, threshold=128):
    img = Image.open(image_path).convert('L')
    mask = np.array(img) > threshold
    return mask.astype(np.uint8)

def close_mask(mask, kernel_size=CLOSE_KERNEL_SIZE):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def bfs_path(mask, start, goal):
    h, w = mask.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and mask[start] == 1):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w and mask[goal] == 1):
        return None
    visited = {start: None}
    q = deque([start])
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            path = []
            cur = (y, x)
            while cur is not None:
                path.append(cur)
                cur = visited[cur]
            return path[::-1]
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1 and (ny, nx) not in visited:
                visited[(ny, nx)] = (y, x)
                q.append((ny, nx))
    return None

def nearest_road_point(mask, x, y, max_dist=100):
    h, w = mask.shape
    for d in range(1, max_dist):
        for dy in range(-d, d+1):
            for dx in range(-d, d+1):
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1:
                    return (ny, nx)
    return None

def on_key(event):
    global start_point, end_point, ax, fig, mask, img, line_path
    if event.key not in ('z', 'x'):
        return
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
    else:
        x = int((ax.get_xlim()[0] + ax.get_xlim()[1]) / 2)
        y = int((ax.get_ylim()[0] + ax.get_ylim()[1]) / 2)
    road_pixel = nearest_road_point(mask, x, y)
    if road_pixel is None:
        print(f"В радиусе 100 пикселей от ({x},{y}) нет дороги")
        return
    if event.key == 'z':
        start_point = road_pixel
        print(f"Старт: {start_point[::-1]}")
        for artist in ax.lines + ax.collections:
            if getattr(artist, '_is_start', False):
                artist.remove()
        start_marker, = ax.plot(start_point[1], start_point[0], 'go', markersize=8, label='Start')
        start_marker._is_start = True
        ax.legend()
    elif event.key == 'x':
        end_point = road_pixel
        print(f"Финиш: {end_point[::-1]}")
        for artist in ax.lines + ax.collections:
            if getattr(artist, '_is_end', False):
                artist.remove()
        end_marker, = ax.plot(end_point[1], end_point[0], 'ro', markersize=8, label='End')
        end_marker._is_end = True
        ax.legend()

    if start_point is not None and end_point is not None:
        path = bfs_path(mask, start_point, end_point)
        if path is None:
            print("Путь не найден (разрывы). Попробуйте увеличить CLOSE_KERNEL_SIZE.")
            return
        if line_path is not None:
            line_path.remove()
        ys, xs = zip(*path)
        line_path, = ax.plot(xs, ys, color='lime', linewidth=2, label='Path')
        line_path._is_path = True
        ax.legend()
        fig.canvas.draw()
        print(f"Путь найден! Длина: {len(path)} шагов")

# ---------- Main ----------
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"Изображение не найдено: {IMAGE_PATH}")
        sys.exit(1)

    # Загружаем маску дорог
    if MODEL_AVAILABLE and os.path.exists(MODEL_PATH):
        print(f"Загружаем модель из {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        if model is None:
            mask = load_road_mask_from_image(IMAGE_PATH)
        else:
            mask = predict_road_mask(model, IMAGE_PATH)
    else:
        mask = load_road_mask_from_image(IMAGE_PATH)

    # Применяем замыкание
    mask_before = mask.copy()
    mask = close_mask(mask)
    print(f"Доля дорог до замыкания: {np.mean(mask_before):.3f}, после: {np.mean(mask):.3f}")

    fig, ax = plt.subplots(figsize=(10, 10))
    img = Image.open(IMAGE_PATH).convert('RGB')
    ax.imshow(img)
    ax.set_title("Наведите курсор, нажмите Z – старт, X – финиш")
    ax.axis('off')

    start_point = None
    end_point = None
    line_path = None

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()