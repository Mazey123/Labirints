"""
Генерация датасета с толстыми связными дорогами.
- Рисует линии толщиной 2 пикселя
- Применяет морфологическое замыкание ко всей карте
- Вырезает фрагменты 512×512
"""

import os
import random
import numpy as np
import osmnx as ox
from shapely.geometry import LineString
from PIL import Image
from tqdm import tqdm
import cv2

# ---------- Конфигурация ----------
OSM_FILE = "Hometown.osm"
OUTPUT_ROOT = "data/real_maps_thick"   # новая папка
TRAIN_IMG_DIR = os.path.join(OUTPUT_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(OUTPUT_ROOT, "train", "masks")
VAL_IMG_DIR = os.path.join(OUTPUT_ROOT, "val", "images")
VAL_MASK_DIR = os.path.join(OUTPUT_ROOT, "val", "masks")

FULL_MAP_SIZE = 2048
CROP_SIZE = 512                     # увеличенный размер
NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 200
ALLOWED_HIGHWAY_TYPES = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'service']
LINE_THICKNESS = 2                  # толщина линий (2 или 3)
CLOSE_KERNEL_SIZE = 3               # замыкание для устранения разрывов (можно увеличить до 5)

# ----------------------------------

def draw_line_on_matrix(matrix, x0, y0, x1, y1, value=0, thickness=1):
    """Рисует линию заданной толщины."""
    h, w = matrix.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        # Закрашиваем квадрат толщины вокруг текущей точки
        for dy_off in range(-thickness//2, thickness//2 + 1):
            for dx_off in range(-thickness//2, thickness//2 + 1):
                ny, nx = y0 + dy_off, x0 + dx_off
                if 0 <= ny < h and 0 <= nx < w:
                    matrix[ny, nx] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def rasterize_full_map(graph, img_size=FULL_MAP_SIZE, thickness=LINE_THICKNESS):
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
    if gdf_nodes.empty or gdf_edges.empty:
        raise ValueError("Граф пуст")

    def normalize_highway(val):
        if isinstance(val, list):
            return val[0] if val else None
        return val
    if 'highway' in gdf_edges.columns:
        gdf_edges['highway'] = gdf_edges['highway'].apply(normalize_highway)
        if ALLOWED_HIGHWAY_TYPES:
            mask = gdf_edges['highway'].isin(ALLOWED_HIGHWAY_TYPES)
            gdf_edges = gdf_edges[mask].copy()
            if gdf_edges.empty:
                raise ValueError("После фильтрации нет рёбер")

    min_x, min_y = gdf_nodes['x'].min(), gdf_nodes['y'].min()
    max_x, max_y = gdf_nodes['x'].max(), gdf_nodes['y'].max()
    padding_geo = 0.02
    dx_geo = max_x - min_x
    dy_geo = max_y - min_y
    min_x -= dx_geo * padding_geo
    max_x += dx_geo * padding_geo
    min_y -= dy_geo * padding_geo
    max_y += dy_geo * padding_geo

    def geo_to_pixel(x, y):
        px = int((x - min_x) / (max_x - min_x) * (img_size - 1))
        py = int((y - min_y) / (max_y - min_y) * (img_size - 1))
        py = img_size - 1 - py
        return px, py

    maze = np.ones((img_size, img_size), dtype=np.uint8)   # 1=стена, 0=дорога
    for _, row in gdf_edges.iterrows():
        geom = row['geometry']
        if not isinstance(geom, LineString):
            continue
        coords = list(geom.coords)
        for i in range(len(coords)-1):
            x0_geo, y0_geo = coords[i]
            x1_geo, y1_geo = coords[i+1]
            x0, y0 = geo_to_pixel(x0_geo, y0_geo)
            x1, y1 = geo_to_pixel(x1_geo, y1_geo)
            draw_line_on_matrix(maze, x0, y0, x1, y1, 0, thickness=thickness)
    return maze

def close_mask(mask, kernel_size=CLOSE_KERNEL_SIZE):
    """Морфологическое замыкание (заполняет мелкие дыры)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def extract_random_crops(maze, num_crops, crop_size=CROP_SIZE):
    h, w = maze.shape
    crops = []
    for _ in range(num_crops):
        y = random.randint(0, h - crop_size)
        x = random.randint(0, w - crop_size)
        crop = maze[y:y+crop_size, x:x+crop_size]
        crops.append(crop)
    return crops

def save_pair(img, mask, img_path, mask_path):
    # img: 0=дорога, 1=стена
    Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
    # маска: дороги = 255, стены = 0
    road_mask = (img == 0).astype(np.uint8) * 255
    Image.fromarray(road_mask).save(mask_path)

def main():
    print("Загрузка графа OSM...")
    G = ox.graph_from_xml(OSM_FILE)
    print("Растеризация всей карты (толстые линии)...")
    full_maze = rasterize_full_map(G, img_size=FULL_MAP_SIZE, thickness=LINE_THICKNESS)
    print("Применяем замыкание для устранения разрывов...")
    full_maze = close_mask(full_maze, kernel_size=CLOSE_KERNEL_SIZE)
    print(f"Доля дорог после замыкания: {np.mean(full_maze == 0):.3f}")

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_MASK_DIR, exist_ok=True)

    print("Вырезание train фрагментов...")
    train_crops = extract_random_crops(full_maze, NUM_TRAIN_SAMPLES, CROP_SIZE)
    for i, crop in enumerate(tqdm(train_crops, desc="Saving train")):
        save_pair(crop, crop,
                  os.path.join(TRAIN_IMG_DIR, f"sample_{i:05d}.png"),
                  os.path.join(TRAIN_MASK_DIR, f"sample_{i:05d}.png"))

    print("Вырезание val фрагментов...")
    val_crops = extract_random_crops(full_maze, NUM_VAL_SAMPLES, CROP_SIZE)
    for i, crop in enumerate(tqdm(val_crops, desc="Saving val")):
        save_pair(crop, crop,
                  os.path.join(VAL_IMG_DIR, f"sample_{i:05d}.png"),
                  os.path.join(VAL_MASK_DIR, f"sample_{i:05d}.png"))

    print(f"Сохранено {len(train_crops)} train и {len(val_crops)} val пар.")
    print("\nТеперь обновите real_map_config.py:")
    print(f"  DATA_ROOT = '{OUTPUT_ROOT}'")
    print(f"  IMG_SIZE = {CROP_SIZE}")
    print("И запустите train_real_maps.py заново.")

if __name__ == "__main__":
    main()