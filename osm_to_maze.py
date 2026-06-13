import os
import numpy as np
import osmnx as ox
from shapely.geometry import LineString
from PIL import Image

# ========== КОНФИГУРАЦИЯ ==========
OSM_FILE = "Town.osm"
OUTPUT_DIR = "data/real_maps"
IMG_SIZE = 2048

# Список разрешённых типов highway. Пустой список = все дороги.
''' Типы:
  - 'footway'
  - 'path'
  - 'pedestrian'
  - 'primary'
  - 'residential'
  - 'secondary'
  - 'service'
  - 'steps'
  - 'tertiary'
'''
ALLOWED_HIGHWAY_TYPES = []

# ===================================

def draw_line_on_matrix(matrix, x0, y0, x1, y1, value=0):
    h, w = matrix.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            matrix[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def rasterize_osm(graph):
    print("--- Преобразование OSM в растр ---")
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

    if gdf_nodes.empty or gdf_edges.empty:
        print("Граф пуст, возвращаем стены.")
        return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Фильтрация по highway
    if ALLOWED_HIGHWAY_TYPES:
        def is_allowed(highway_val):
            if isinstance(highway_val, list):
                return any(h in ALLOWED_HIGHWAY_TYPES for h in highway_val)
            elif isinstance(highway_val, str):
                return highway_val in ALLOWED_HIGHWAY_TYPES
            else:
                return False
        
        mask = gdf_edges['highway'].apply(is_allowed)
        filtered_edges = gdf_edges[mask].copy()
        print(f"  Всего рёбер: {len(gdf_edges)}, отобрано: {len(filtered_edges)}")
        if filtered_edges.empty:
            print("  После фильтрации нет рёбер. Выход.")
            return np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        gdf_edges = filtered_edges
    else:
        print("  Фильтр не задан – все дороги.")

    # Границы
    min_x, min_y = gdf_nodes['x'].min(), gdf_nodes['y'].min()
    max_x, max_y = gdf_nodes['x'].max(), gdf_nodes['y'].max()

    padding = 0.02
    dx_geo = max_x - min_x
    dy_geo = max_y - min_y
    min_x -= dx_geo * padding
    max_x += dx_geo * padding
    min_y -= dy_geo * padding
    max_y += dy_geo * padding

    def geo_to_pixel(x, y):
        px = int((x - min_x) / (max_x - min_x) * (IMG_SIZE - 1))
        py = int((y - min_y) / (max_y - min_y) * (IMG_SIZE - 1))
        py = IMG_SIZE - 1 - py
        return px, py

    maze = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    total = len(gdf_edges)

    for idx, (_, row) in enumerate(gdf_edges.iterrows()):
        geom = row['geometry']
        if not isinstance(geom, LineString):
            continue
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            x0_geo, y0_geo = coords[i]
            x1_geo, y1_geo = coords[i+1]
            x0, y0 = geo_to_pixel(x0_geo, y0_geo)
            x1, y1 = geo_to_pixel(x1_geo, y1_geo)
            draw_line_on_matrix(maze, x0, y0, x1, y1, 0)
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"  Прогресс: {idx+1}/{total} рёбер")

    print("--- Преобразование завершено ---")
    return maze

def save_results(maze, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, maze)
    img = Image.fromarray((maze * 255).astype(np.uint8))
    png_path = output_path.replace('.npy', '.png')
    img.save(png_path)
    print(f"✅ Сохранено: {png_path}")

if __name__ == "__main__":
    if not os.path.exists(OSM_FILE):
        print(f"Файл {OSM_FILE} не найден.")
    else:
        try:
            G = ox.graph_from_xml(OSM_FILE)
            maze = rasterize_osm(G)
            save_results(maze, os.path.join(OUTPUT_DIR, "maze.npy"))
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()