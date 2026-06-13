import osmnx as ox
import pandas as pd

OSM_FILE = "Hometown.osm"

# Загружаем граф
G = ox.graph_from_xml(OSM_FILE)

# Получаем рёбра
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# Функция, которая приводит highway к набору строк (извлекает все уникальные значения)
def extract_highway_values(series):
    values = set()
    for val in series.dropna():
        if isinstance(val, list):
            # Если список, добавляем каждый элемент
            values.update(val)
        elif isinstance(val, str):
            values.add(val)
        # игнорируем другие типы (None, float)
    return sorted(values)

unique_highway = extract_highway_values(gdf_edges['highway'])
print("Все уникальные типы highway в файле:")
for h in unique_highway:
    print(f"  - {h}")