"""
Генератор лабиринтов с GUI, выбором сложности, алгоритмов генерации
и интеграцией с нейросетью для предсказания пути.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tkinter as tk
from tkinter import ttk
import os
import uuid
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
MAZES_FOLDER = "mazes"
SOLUTIONS_FOLDER = "solutions"
DEFAULT_MODEL_PATH = "maze_model_best_f1.pth"

class Difficulty(Enum):
    EASY = 'легкий'
    MEDIUM = 'средний'
    HARD = 'сложный'

class Algorithm(Enum):
    PRIM = 'Prim'
    KRUSKAL = 'Kruskal'
    ELLER = 'Eller'
    ALDOUS_BRODER = 'Aldous-Бroдер'

@dataclass
class MazeConfig:
    """Конфигурация для генерации лабиринта."""
    difficulty: Difficulty
    algorithm: Algorithm
    force_unsolvable: bool = False
    show_path: bool = False
    show_nn_path: bool = False
    max_attempts: int = 10

@dataclass
class MazeResult:
    """Результат генерации лабиринта."""
    maze: np.ndarray
    entry: Optional[Tuple[int, int]] = None
    exit: Optional[Tuple[int, int]] = None
    path: Optional[List[Tuple[int, int]]] = None
    nn_path_mask: Optional[np.ndarray] = None
    is_solvable: bool = False

# Функция выбора размера по уровню сложности
def get_maze_size(difficulty) -> Tuple[int, int]:
    """Возвращает размеры лабиринта berdasarkan уровня сложности."""
    size_map = {
        'легкий': (33, 33),
        'средний': (55, 55),
        'сложный': (77, 77),
        Difficulty.EASY: (33, 33),
        Difficulty.MEDIUM: (55, 55),
        Difficulty.HARD: (77, 77),
    }
    return size_map.get(difficulty, (33, 33))


def is_solvable(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Проверяет проходимость лабиринта и возвращает путь, если он существует.
    
    Args:
        maze: Двумерный массив лабиринта (0 - проход, 1 - стена)
        start: Координаты начала (y, x)
        end: Координаты конца (y, x)
    
    Returns:
        Кортеж из (проходимость, путь)
    """
    if not isinstance(maze, np.ndarray) or maze.ndim != 2:
        logger.error("Некорректный формат лабиринта")
        return False, []
    
    visited = {start: None}
    queue = deque([start])
    
    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            # Восстанавливаем путь
            path = []
            current = (y, x)
            while current is not None:
                path.append(current)
                current = visited[current]
            return True, path[::-1]
        
        # Проверяем соседние клетки
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1] and 
                maze[ny, nx] == 0 and (ny, nx) not in visited):
                visited[(ny, nx)] = (y, x)
                queue.append((ny, nx))
    
    return False, []


def save_maze(maze: np.ndarray, algorithm: str, folder: str = MAZES_FOLDER) -> Optional[str]:
    """Сохраняет лабиринт в файл."""
    try:
        os.makedirs(folder, exist_ok=True)
        filename = f"maze_{algorithm}_{uuid.uuid4().hex[:8]}.npy"
        filepath = os.path.join(folder, filename)
        np.save(filepath, maze)
        logger.info(f"Лабиринт сохранён: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Ошибка при сохранении лабиринта: {e}")
        return None

# --- Алгоритмы генерации ---

def generate_maze_prim(width: int, height: int) -> np.ndarray:
    """
    Генерация лабиринта алгоритмом Прима.
    
    Args:
        width: Ширина лабиринта
        height: Высота лабиринта
    
    Returns:
        Двумерный массив лабиринта
    """
    maze = np.ones((height, width), dtype=np.uint8)
    start_y, start_x = 1, 1
    maze[start_y, start_x] = 0
    
    walls = []
    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        ny, nx = start_y + dy, start_x + dx
        if 0 <= ny < height and 0 <= nx < width:
            walls.append((ny, nx, start_y + dy // 2, start_x + dx // 2))
    
    while walls:
        idx = random.randint(0, len(walls) - 1)
        y, x, wy, wx = walls.pop(idx)
        if maze[y, x] == 1:
            maze[wy, wx] = 0
            maze[y, x] = 0
            for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 1:
                    walls.append((ny, nx, y + dy // 2, x + dx // 2))
    
    return maze


def generate_maze_kruskal(width: int, height: int) -> np.ndarray:
    """
    Генерация лабиринта алгоритмом Краскала.
    
    Args:
        width: Ширина лабиринта
        height: Высота лабиринта
    
    Returns:
        Двумерный массив лабиринта
    """
    maze = np.ones((height, width), dtype=np.uint8)
    sets = {}
    cells = []
    set_id = 1
    
    for y in range(1, height, 2):
        for x in range(1, width, 2):
            maze[y, x] = 0
            sets[(y, x)] = set_id
            set_id += 1
            cells.append((y, x))
    
    walls = []
    for y, x in cells:
        for dy, dx in [(0, 2), (2, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                walls.append(((y, x), (ny, nx), (y + dy // 2, x + dx // 2)))
    
    random.shuffle(walls)
    
    for (y1, x1), (y2, x2), (wy, wx) in walls:
        if sets[(y1, x1)] != sets[(y2, x2)]:
            maze[wy, wx] = 0
            old_set, new_set = sets[(y2, x2)], sets[(y1, x1)]
            for key in sets:
                if sets[key] == old_set:
                    sets[key] = new_set
    
    return maze


def generate_maze_eller(width: int, height: int) -> Optional[np.ndarray]:
    """
    Генерация лабиринта алгоритмом Эллера.
    
    Args:
        width: Ширина лабиринта (должна быть нечётной)
        height: Высота лабиринта (должна быть нечётной)
    
    Returns:
        Двумерный массив лабиринта или None при ошибке
    """
    if width % 2 == 0 or height % 2 == 0:
        logger.warning("Алгоритм Эллера требует нечётных размеров")
        return None
    
    maze = np.ones((height, width), dtype=np.uint8)
    sets = [0] * (width // 2)
    next_set = 1
    
    for y in range(1, height - 2, 2):
        # Присваиваем множества клеткам без множества
        for x in range(width // 2):
            if sets[x] == 0:
                sets[x] = next_set
                next_set += 1
        
        # Соединяем соседние клетки в строке
        for x in range(width // 2 - 1):
            if sets[x] != sets[x + 1] and random.choice([True, False]):
                maze[y, 2 * x + 2] = 0
                old_set = sets[x + 1]
                for i in range(len(sets)):
                    if sets[i] == old_set:
                        sets[i] = sets[x]
        
        # Создаём вертикальные проходы вниз
        below = [False] * (width // 2)
        unique_sets = set(sets)
        for s in unique_sets:
            indices = [i for i, v in enumerate(sets) if v == s]
            random.shuffle(indices)
            num_down = random.randint(1, len(indices))
            for i in indices[:num_down]:
                maze[y + 1, 2 * i + 1] = 0
                below[i] = True
        
        # Очищаем множества для клеток без прохода вниз
        for x in range(width // 2):
            if not below[x]:
                sets[x] = 0
    
    # Последняя строка: соединяем все множества
    y = height - 2
    for x in range(width // 2):
        if sets[x] == 0:
            sets[x] = next_set
            next_set += 1
    
    for x in range(width // 2 - 1):
        if sets[x] != sets[x + 1]:
            maze[y, 2 * x + 2] = 0
            old_set = sets[x + 1]
            for i in range(len(sets)):
                if sets[i] == old_set:
                    sets[i] = sets[x]
    
    # Заполняем все клетки на нечётных позициях как проходы
    for y in range(1, height, 2):
        for x in range(1, width, 2):
            maze[y, x] = 0
    
    return maze


def generate_maze_aldous_broder(width: int, height: int, max_steps_factor: int = 100) -> Optional[np.ndarray]:
    """
    Генерация лабиринта алгоритмом Aldous-Broder.
    
    Args:
        width: Ширина лабиринта
        height: Высота лабиринта
        max_steps_factor: Множитель для ограничения шагов
    
    Returns:
        Двумерный массив лабиринта или None при неудаче
    """
    maze = np.ones((height, width), dtype=np.uint8)
    
    # Выбираем случайную стартовую клетку
    start_y = random.randrange(1, height, 2)
    start_x = random.randrange(1, width, 2)
    maze[start_y, start_x] = 0
    
    visited = {(start_y, start_x)}
    total_cells = ((height - 1) // 2) * ((width - 1) // 2)
    max_steps = width * height * max_steps_factor
    
    y, x = start_y, start_x
    steps = 0
    
    while len(visited) < total_cells and steps < max_steps:
        dy, dx = random.choice([(-2, 0), (2, 0), (0, -2), (0, 2)])
        ny, nx = y + dy, x + dx
        
        if 1 <= ny < height - 1 and 1 <= nx < width - 1:
            if (ny, nx) not in visited:
                maze[y + dy // 2, x + dx // 2] = 0
                maze[ny, nx] = 0
                visited.add((ny, nx))
            y, x = ny, nx
        
        steps += 1
    
    if len(visited) < total_cells:
        logger.warning(
            f"Aldous-Broder: не удалось сгенерировать лабиринт "
            f"(посещено {len(visited)}/{total_cells} клеток за {steps} шагов)"
        )
        return None
    
    return maze


# --- Основная функция генерации с выбором алгоритма ---

def find_edges(maze: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    """Находит все возможные точки входа/выхода на краях лабиринта."""
    height, width = maze.shape
    edges = {
        'top': [(0, x) for x in range(1, width, 2) if maze[1, x] == 0],
        'bottom': [(height - 1, x) for x in range(1, width, 2) if maze[height - 2, x] == 0],
        'left': [(y, 0) for y in range(1, height, 2) if maze[y, 1] == 0],
        'right': [(y, width - 1) for y in range(1, height, 2) if maze[y, width - 2] == 0],
    }
    return edges


def generate_maze(
    difficulty: str = 'средний',
    force_unsolvable: bool = False,
    max_attempts: int = 10,
    algorithm_name: Optional[str] = None,
    show_path: bool = False,
    show_nn_path: bool = False
) -> Optional[MazeResult]:
    """
    Генерирует лабиринт с заданными параметрами.
    
    Args:
        difficulty: Уровень сложности ('легкий', 'средний', 'сложный')
        force_unsolvable: Если True, создаёт непроходимый лабиринт
        max_attempts: Максимальное количество попыток генерации
        algorithm_name: Название алгоритма генерации
        show_path: Показать путь BFS
        show_nn_path: Показать путь от нейросети
    
    Returns:
        MazeResult с результатами генерации или None при ошибке
    """
    width, height = get_maze_size(difficulty)
    
    # Определяем алгоритм
    if algorithm_name is None:
        # Для GUI берём из комбобокса
        try:
            algorithm_name = algo_combo.get()
        except NameError:
            algorithm_name = 'Prim'
    
    min_path_length = (width + height) // 2
    
    # Сопоставление названий алгоритмов
    algo_mapping = {
        'Prim (default)': ('prim', generate_maze_prim),
        'Prim': ('prim', generate_maze_prim),
        'Kruskal': ('kruskal', generate_maze_kruskal),
        'Eller': ('eller', generate_maze_eller),
        'Aldous-Бroдер': ('aldous_broder', generate_maze_aldous_broder),
    }
    
    if algorithm_name not in algo_mapping:
        logger.error(f"Неизвестный алгоритм: {algorithm_name}")
        return None
    
    algo_name, algo_func = algo_mapping[algorithm_name]
    
    for attempt in range(max_attempts):
        # Генерируем лабиринт
        maze = algo_func(width, height)
        if maze is None:
            logger.warning(f"Попытка {attempt + 1}: генерация не удалась")
            continue
        
        # Находим края для входа/выхода
        edges = find_edges(maze)
        
        if not any(edges.values()):
            continue
        
        # Пытаемся найти такие вход и выход, чтобы путь был достаточно длинным
        edge_keys = [k for k, v in edges.items() if v]
        found = False
        entry_point = None
        exit_point = None
        path = []
        
        for _ in range(20):  # 20 попыток подобрать пару
            if len(edge_keys) < 2:
                break
                
            entry_edge = random.choice(edge_keys)
            exit_edge_candidates = [e for e in edge_keys if e != entry_edge]
            
            if not exit_edge_candidates:
                continue
            
            exit_edge = random.choice(exit_edge_candidates)
            entry_candidate = random.choice(edges[entry_edge])
            exit_candidate = random.choice(edges[exit_edge])
            
            # Временно открываем вход и выход
            maze[entry_candidate] = 0
            maze[exit_candidate] = 0
            
            is_path_found, candidate_path = is_solvable(maze, entry_candidate, exit_candidate)
            
            if is_path_found and len(candidate_path) >= min_path_length:
                entry_point = entry_candidate
                exit_point = exit_candidate
                path = candidate_path
                found = True
                break
            
            # Закрываем обратно
            maze[entry_candidate] = 1
            maze[exit_candidate] = 1
        
        if not found:
            continue
        
        # Проверяем проходимость
        is_solvable_result, path = is_solvable(maze, entry_point, exit_point)
        
        if force_unsolvable:
            if not is_solvable_result:
                # Уже непроходимый
                break
            # Блокируем путь посередине
            mid_idx = len(path) // 2
            mid_y, mid_x = path[mid_idx]
            maze[max(0, mid_y-1):min(height, mid_y+2), max(0, mid_x-1):min(width, mid_x+2)] = 1
            
            if not is_solvable(maze, entry_point, exit_point)[0]:
                break
        else:
            if is_solvable_result:
                break
    
    # Создаём результат
    result = MazeResult(
        maze=maze,
        entry=entry_point,
        exit=exit_point,
        path=path if is_solvable_result else [],
        is_solvable=is_solvable_result
    )
    
    # Предсказание пути нейросетью (если запрошено)
    nn_path_mask = None
    if is_solvable_result and show_nn_path:
        try:
            nn_path_mask = predict_solution(maze)
            result.nn_path_mask = nn_path_mask
        except Exception as e:
            logger.error(f"Ошибка предсказания нейросети: {e}")
    
    # Отображение
    if show_path or show_nn_path:
        if not force_unsolvable and is_solvable_result:
            show_maze_visual(path if show_path else None, nn_path_mask if show_nn_path else None)
        elif show_nn_path and is_solvable_result:
            show_maze_visual(None, nn_path_mask)
        else:
            show_maze_visual()
    
    return result


def show_maze_visual(path: Optional[List[Tuple[int, int]]] = None, nn_path_mask: Optional[np.ndarray] = None):
    """
    Отображает лабиринт с опциональными путями.
    
    Args:
        path: Путь от BFS алгоритма
        nn_path_mask: Маска пути от нейросети
    """
    global maze
    
    if not isinstance(maze, np.ndarray):
        logger.error("maze не определён или не является numpy массивом")
        return
    
    if maze.ndim != 2:
        logger.error(f"Некорректная размерность maze: {maze.shape}")
        plt.close()
        return
    
    plt.figure(figsize=(10, 7))
    plt.imshow(maze, cmap='Greys')
    
    # Отображение пути BFS
    if path:
        for (y1, x1), (y2, x2) in zip(path[:-1], path[1:]):
            plt.plot([x1, x2], [y1, y2], color='green', linewidth=2, 
                    label='BFS путь' if (y1, x1) == path[0] else "")
    
    # Отображение пути нейросети
    if nn_path_mask is not None:
        if nn_path_mask.ndim == 2:
            plt.imshow(nn_path_mask, cmap='Reds', alpha=0.4)
    
    # Легенда
    if path and nn_path_mask is not None:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='BFS путь'),
            Line2D([0], [0], color='red', lw=6, alpha=0.4, label='Путь нейросети')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Лабиринт")
    plt.axis('off')
    plt.show()

# Функция для пакетной генерации лабиринтов
def batch_generate_mazes(difficulty, count_per_algo=10000):
    algos = [
        ('Prim (default)', 'prim'),
        ('Kruskal', 'kruskal'),
        ('Eller', 'eller'),
        ('Aldous-Broдер', 'aldous_broder')
    ]
    os.makedirs("mazes", exist_ok=True)
    os.makedirs("solutions", exist_ok=True)
    for algo_gui, algo_name in algos:
        print(f"Генерация {count_per_algo} лабиринтов для алгоритма {algo_name}...")
        for i in range(count_per_algo):
            width, height = get_maze_size(difficulty)
            # Генерируем лабиринт
            if algo_gui == 'Prim (default)':
                maze = generate_maze_prim(width, height)
            elif algo_gui == 'Kruskal':
                maze = generate_maze_kruskal(width, height)
            elif algo_gui == 'Eller':
                maze = generate_maze_eller(width, height)
            elif algo_gui == 'Aldous-Бroдер':
                maze = generate_maze_aldous_broder(width, height)

            # --- Добавляем вход и выход ---
            edges = {
                'top': [(0, x) for x in range(1, width, 2) if maze[1][x] == 0],
                'bottom': [(height - 1, x) for x in range(1, width, 2) if maze[height - 2][x] == 0],
                'left': [(y, 0) for y in range(1, height, 2) if maze[y][1] == 0],
                'right': [(y, width - 1) for y in range(1, height, 2) if maze[y][width - 2] == 0],
            }
            edge_keys = [k for k, v in edges.items() if v]
            if len(edge_keys) >= 2:
                entry_edge = random.choice(edge_keys)
                exit_edge_candidates = [e for e in edge_keys if e != entry_edge]
                exit_edge = random.choice(exit_edge_candidates)
                entry_candidate = random.choice(edges[entry_edge])
                exit_candidate = random.choice(edges[exit_edge])
                maze[entry_candidate] = 0
                maze[exit_candidate] = 0
            # --- конец добавления входа/выхода ---

            # --- Находим решение и сохраняем ---
            # Поиск входов/выходов (повторно, чтобы быть уверенным)
            entries = []
            exits = []
            for x in range(width):
                if maze[0, x] == 0:
                    entries.append((0, x))
                if maze[height-1, x] == 0:
                    exits.append((height-1, x))
            for y in range(height):
                if maze[y, 0] == 0:
                    entries.append((y, 0))
                if maze[y, width-1] == 0:
                    exits.append((y, width-1))
            found = False
            path = []
            # Корректно ищем путь между всеми парами (entry, exit)
            for entry in entries:
                for exit in exits:
                    solvable, candidate_path = is_solvable(maze, entry, exit)
                    if solvable:
                        found = True
                        path = candidate_path
                        break
                if found:
                    break
            if found and path:
                filename = f"maze_{algo_name}_{uuid.uuid4().hex[:8]}.npy"
                np.save(os.path.join("mazes", filename), maze)
                path_mask = np.zeros_like(maze, dtype=np.uint8)
                for y_, x_ in path:
                    path_mask[y_, x_] = 1
                np.save(os.path.join("solutions", filename), path_mask)
            # иначе не сохраняем!
            if (i+1) % 1000 == 0:
                print(f"{i+1} сгенерировано для {algo_name}")

def solve_and_save_all_mazes(mazes_folder="mazes", solutions_folder="solutions"):
    """
    Для каждого лабиринта из папки mazes находит путь (если есть) и сохраняет его в solutions.
    """
    os.makedirs(solutions_folder, exist_ok=True)
    maze_files = [f for f in os.listdir(mazes_folder) if f.endswith('.npy')]
    for fname in maze_files:
        maze = np.load(os.path.join(mazes_folder, fname))
        # Найти вход и выход (по краям, где maze==0)
        height, width = maze.shape
        entries = []
        exits = []
        for x in range(width):
            if maze[0, x] == 0:
                entries.append((0, x))
            if maze[height-1, x] == 0:
                exits.append((height-1, x))
        for y in range(height):
            if maze[y, 0] == 0:
                entries.append((y, 0))
            if maze[y, width-1] == 0:
                exits.append((y, width-1))
        if not entries or not exits:
            continue
        found = False
        for entry in entries:
            for exit in exits:
                solvable, path = is_solvable(maze, entry, exit)
                if solvable:
                    found = True
                    break
            if found:
                break
        if found and path:
            path_mask = np.zeros_like(maze, dtype=np.uint8)
            for y, x in path:
                path_mask[y, x] = 1
            np.save(os.path.join(solutions_folder, fname), path_mask)
        else:
            np.save(os.path.join(solutions_folder, fname), np.zeros_like(maze, dtype=np.uint8))

class SimpleMazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

# 4. Проверка наличия файла модели
def predict_solution(maze_np, model_path="maze_model_best_f1.pth"):
    if not os.path.exists(model_path):
        print("Файл модели не найден.")
        return np.zeros_like(maze_np)
    # --- определяем класс модели для загрузки ---
    from train_maze_model import UNet  # Импортируйте UNet из вашего файла обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # --- инверсия, если обучали на инвертированных лабиринтах ---
    maze_input = 1 - maze_np  # если при обучении была инверсия!
    maze_tensor = torch.tensor(maze_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(maze_tensor)
        prob_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        if prob_mask.shape != maze_np.shape:
            print("Внимание: размер лабиринта не совпадает с размерами, на которых обучалась нейросеть. Результат может быть некорректным.")
        print("prob_mask min:", prob_mask.min(), "max:", prob_mask.max(), "mean:", prob_mask.mean())
        # --- попробуем показать heatmap вероятностей ---
        plt.figure(figsize=(8, 6))
        plt.title("Heatmap вероятности пути (output sigmoid)")
        plt.imshow(prob_mask, cmap='hot')
        plt.colorbar()
        plt.show()
        # --- попробуем ещё ниже порог ---
        threshold = 0.05  # очень низкий порог для визуализации
        from scipy.ndimage import label
        labeled, num = label(prob_mask > threshold)
        if num > 0:
            sizes = [(labeled == i).sum() for i in range(1, num+1)]
            max_label = np.argmax(sizes) + 1
            path_mask = (labeled == max_label).astype(np.uint8)
        else:
            path_mask = (prob_mask > threshold).astype(np.uint8)
        print("Количество клеток, предсказанных как путь:", np.sum(path_mask))
    return path_mask

def show_predicted_path(maze, path_mask):
    plt.figure(figsize=(10,7))
    plt.imshow(maze, cmap='binary')
    plt.imshow(path_mask, cmap='Reds', alpha=0.5)
    plt.title("Лабиринт с предсказанным путём")
    plt.axis('off')
    plt.show()

def check_solutions_folder(solutions_folder="solutions"):
    """
    Проверяет, сколько клеток пути содержится в каждом файле solutions.
    """
    files = [f for f in os.listdir(solutions_folder) if f.endswith('.npy')]
    for fname in files:
        arr = np.load(os.path.join(solutions_folder, fname))
        path_cells = np.sum(arr)
        print(f"{fname}: путь содержит {path_cells} клеток")

if __name__ == "__main__":
    # Для запуска пакетной генерации раскомментируйте строку ниже:
    #batch_generate_mazes('сложный', 30000)
    #exit()

    # GUI запускать только если не идет пакетная генерация
    root = tk.Tk()
    root.title("Настройка генерации лабиринта")

    label = ttk.Label(root, text="Выберите уровень сложности:")
    label.pack(padx=10, pady=5)

    combo = ttk.Combobox(root, values=['легкий', 'средний', 'сложный'], state='readonly')
    combo.current(0)
    combo.pack(padx=10, pady=5)

    # Новый выпадающий список для выбора алгоритма
    algo_label = ttk.Label(root, text="Алгоритм генерации:")
    algo_label.pack(padx=10, pady=5)
    algo_combo = ttk.Combobox(root, values=[
        'Prim', 'Kruskal', 'Eller', 'Aldous-Бroдер'
    ], state='readonly')
    algo_combo.current(0)
    algo_combo.pack(padx=10, pady=5)

    check_var = tk.BooleanVar(value=False)
    check = ttk.Checkbutton(root, text="Гарантировать непроходимость", variable=check_var)
    check.pack(padx=10, pady=5)

    # Новый чекбокс для отображения пути к выходу
    show_path_var = tk.BooleanVar(value=False)
    show_path_check = ttk.Checkbutton(root, text="Показать путь к выходу", variable=show_path_var)
    show_path_check.pack(padx=10, pady=5)

    # Новый чекбокс для отображения пути, найденного нейросетью
    show_nn_path_var = tk.BooleanVar(value=False)
    show_nn_path_check = ttk.Checkbutton(root, text="Путь от нейросети", variable=show_nn_path_var)
    show_nn_path_check.pack(padx=10, pady=5)

    btn_generate = ttk.Button(root, text="Сгенерировать лабиринт", command=lambda: [generate_maze(combo.get(), check_var.get())])
    btn_generate.pack(padx=10, pady=5)

    # Обновление доступности чекбоксов друг относительно друга
    def update_check_states(*args):
        if check_var.get():
            show_path_check.config(state=tk.DISABLED)
            show_nn_path_check.config(state=tk.DISABLED)
        else:
            show_path_check.config(state=tk.NORMAL)
            show_nn_path_check.config(state=tk.NORMAL)
        if show_path_var.get() or show_nn_path_var.get():
            check.config(state=tk.DISABLED)
        else:
            check.config(state=tk.NORMAL)

    check_var.trace_add('write', update_check_states)
    show_path_var.trace_add('write', update_check_states)
    show_nn_path_var.trace_add('write', update_check_states)
    update_check_states()

    root.mainloop()