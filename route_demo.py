"""
Построение маршрута с принудительным соединением разорванных компонент дорог.
Левая кнопка – старт, правая – финиш.
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
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

# -------------------------------------------------------------------
# НАСТРОЙКИ
# -------------------------------------------------------------------
IMAGE_PATH = "data/real_maps_thick/val/images/sample_00000.png"
MODEL_PATH = "models_real_maps/real_map_best_iou.pth"
IMG_SIZE = 512
THRESHOLD = 0.5
CLOSE_KERNEL = 0        # сильное замыкание для устранения разрывов
CONNECT_DIST = 0       # максимальное расстояние для соединения компонент (пикселей)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------------------------------------------------

sys.path.append(os.path.dirname(__file__))
try:
    from train_maze_model import UNet
except ImportError:
    print("Ошибка: не найден train_maze_model.py")
    sys.exit(1)

def invert_image(x):
    return 1 - x

def load_model(path):
    if not os.path.exists(path):
        print(f"Модель не найдена: {path}")
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

def predict_mask(model, image_path):
    img = Image.open(image_path).convert('L')
    orig_size = img.size
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Lambda(invert_image)
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(input_tensor)
        prob = torch.sigmoid(out).cpu().squeeze().numpy()
    mask_small = (prob > THRESHOLD).astype(np.uint8)
    mask_img = Image.fromarray((mask_small * 255).astype(np.uint8))
    mask_img = mask_img.resize(orig_size, Image.NEAREST)
    return np.array(mask_img) // 255

def connect_components(mask, max_dist=CONNECT_DIST):
    """Соединяет близкие компоненты дорог прямыми линиями."""
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 2:
        return mask
    # Находим центроиды (или любую точку) каждой компоненты
    points = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(ys) == 0:
            continue
        # Ближайшая к центру точка (можно использовать любую)
        centroid = (np.mean(ys).astype(int), np.mean(xs).astype(int))
        # Ищем фактическую точку компоненты, ближайшую к центроиду
        tree = KDTree(np.array(list(zip(ys, xs))))
        dist, idx = tree.query(centroid)
        points.append((xs[idx], ys[idx], label))  # (x, y, label)
    # Соединяем компоненты, если расстояние меньше max_dist
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            x1, y1, lab1 = points[i]
            x2, y2, lab2 = points[j]
            dist = np.hypot(x2-x1, y2-y1)
            if dist < max_dist:
                # Рисуем линию между точками
                cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=2)
    return mask

def bfs_path(mask, start, goal):
    h, w = mask.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and mask[start] == 1):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w and mask[goal] == 1):
        return None
    q = deque([start])
    parent = {start: None}
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            path = []
            cur = (y, x)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1 and (ny, nx) not in parent:
                parent[(ny, nx)] = (y, x)
                q.append((ny, nx))
    return None

def nearest_road(mask, y, x, max_dist=150):
    h, w = mask.shape
    if mask[y, x] == 1:
        return (y, x)
    for d in range(1, max_dist):
        for dy in range(-d, d+1):
            for dx in range(-d, d+1):
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1:
                    return (ny, nx)
    return None

def on_click(event):
    global start_point, end_point, ax, fig, mask, line_path
    if event.inaxes != ax:
        return
    x, y = int(event.xdata), int(event.ydata)
    road = nearest_road(mask, y, x)
    if road is None:
        print(f"Нет дороги рядом с ({x},{y})")
        return
    if event.button == 1:
        start_point = road
        print(f"Старт: {start_point[::-1]}")
        for artist in ax.lines + ax.collections:
            if getattr(artist, '_is_start', False):
                artist.remove()
        marker, = ax.plot(start_point[1], start_point[0], 'go', markersize=8, label='Start')
        marker._is_start = True
        ax.legend()
    elif event.button == 3:
        end_point = road
        print(f"Финиш: {end_point[::-1]}")
        for artist in ax.lines + ax.collections:
            if getattr(artist, '_is_end', False):
                artist.remove()
        marker, = ax.plot(end_point[1], end_point[0], 'ro', markersize=8, label='End')
        marker._is_end = True
        ax.legend()
    if start_point is not None and end_point is not None:
        path = bfs_path(mask, start_point, end_point)
        if path is None:
            print("Путь не найден даже после соединения компонент.")
            return
        if line_path is not None:
            line_path.remove()
        ys, xs = zip(*path)
        line_path, = ax.plot(xs, ys, color='lime', linewidth=2, label='Path')
        line_path._is_path = True
        ax.legend()
        fig.canvas.draw()
        print(f"Путь найден! Длина: {len(path)} шагов")
        plt.savefig("route_result_connected.png", dpi=150, bbox_inches='tight')
        print("Результат сохранён в route_result_connected.png")

def main():
    global start_point, end_point, ax, fig, mask, line_path

    if not os.path.exists(IMAGE_PATH):
        print(f"Изображение не найдено: {IMAGE_PATH}")
        return

    model = load_model(MODEL_PATH)
    if model is None:
        return

    print("Предсказание маски дорог...")
    raw_mask = predict_mask(model, IMAGE_PATH)
    print(f"Доля дорог: {np.mean(raw_mask):.3f}")

    # 1. Сильное замыкание
    kernel = np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    print(f"После замыкания: {np.mean(mask):.3f}")

    # 2. Соединение компонент
    mask = connect_components(mask, max_dist=CONNECT_DIST)
    print(f"После соединения компонент: {np.mean(mask):.3f}")

    # 3. Опционально: дилатация (для надёжности)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 10))
    img = Image.open(IMAGE_PATH).convert('RGB')
    ax.imshow(img)

    # Наложение полупрозрачной маски
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask == 1] = [0, 255, 255, 70]
    ax.imshow(overlay, alpha=0.3)

    ax.set_title("Левый клик – старт, правый – финиш. Путь зелёный.")
    ax.axis('off')

    start_point = None
    end_point = None
    line_path = None

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == "__main__":
    main()