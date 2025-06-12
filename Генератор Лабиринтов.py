# Версия с GUI, выбором сложности, непроходимостью и отображением пути к выходу
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import tkinter as tk
from tkinter import ttk
import os
import uuid
import torch
import torch.nn as nn

maze = None
entry = None
exit = None

# Функция выбора размера по уровню сложности
def get_maze_size(difficulty):
    if difficulty == 'легкий':
        return 33, 33
    elif difficulty == 'средний':
        return 55, 55
    elif difficulty == 'сложный':
        return 77, 77

# Проверка на проходимость
# Также возвращает путь, если он существует
def is_solvable(maze, start, end):
    visited = {}
    queue = deque([start])
    visited[start] = None
    while queue:
        y, x = queue.popleft()
        if (y, x) == end:
            path = []
            current = (y, x)
            while current is not None:
                path.append(current)
                current = visited[current]
            return True, path[::-1]
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]:
                if maze[ny][nx] == 0 and (ny, nx) not in visited:
                    visited[(ny, nx)] = (y, x)
                    queue.append((ny, nx))
    return False, []

def save_maze(maze, algorithm):
    os.makedirs("mazes", exist_ok=True)
    filename = f"mazes/maze_{algorithm}_{uuid.uuid4().hex[:8]}.npy"
    np.save(filename, maze)

# --- Алгоритмы генерации ---

def generate_maze_prim(width, height):
    # Алгоритм Прима (по умолчанию)
    maze = np.ones((height, width), dtype=int)
    start_y, start_x = 1, 1
    maze[start_y][start_x] = 0
    walls = []
    for dy, dx in [(-2,0),(2,0),(0,-2),(0,2)]:
        ny, nx = start_y + dy, start_x + dx
        if 0 <= ny < height and 0 <= nx < width:
            walls.append((ny, nx, start_y + dy//2, start_x + dx//2))
    while walls:
        y, x, wy, wx = walls.pop(random.randint(0, len(walls)-1))
        if maze[y][x] == 1:
            maze[wy][wx] = 0
            maze[y][x] = 0
            for dy, dx in [(-2,0),(2,0),(0,-2),(0,2)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and maze[ny][nx] == 1:
                    walls.append((ny, nx, y + dy//2, x + dx//2))
    return maze

def generate_maze_kruskal(width, height):
    # Алгоритм Краскала
    maze = np.ones((height, width), dtype=int)
    sets = {}
    cells = []
    set_id = 1
    for y in range(1, height, 2):
        for x in range(1, width, 2):
            maze[y][x] = 0
            sets[(y, x)] = set_id
            set_id += 1
            cells.append((y, x))
    walls = []
    for y, x in cells:
        for dy, dx in [(0, 2), (2, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                walls.append(((y, x), (ny, nx), (y + dy//2, x + dx//2)))
    random.shuffle(walls)
    for (y1, x1), (y2, x2), (wy, wx) in walls:
        if sets[(y1, x1)] != sets[(y2, x2)]:
            maze[wy][wx] = 0
            old, new = sets[(y2, x2)], sets[(y1, x1)]
            for k in sets:
                if sets[k] == old:
                    sets[k] = new
    return maze

def generate_maze_eller(width, height):
    # Алгоритм Эллера 
    maze = np.ones((height, width), dtype=int)
    sets = [0] * (width // 2)
    next_set = 1
    for y in range(1, height-2, 2):  
        # Присваиваем множества
        for x in range(width // 2):
            if sets[x] == 0:
                sets[x] = next_set
                next_set += 1
        # Соединяем клетки в строке
        for x in range(width // 2 - 1):
            if sets[x] != sets[x+1] and random.choice([True, False]):
                maze[y][2*x+2] = 0
                old_set = sets[x+1]
                for i in range(len(sets)):
                    if sets[i] == old_set:
                        sets[i] = sets[x]
        # Делаем вертикальные проходы
        below = [False] * (width // 2)
        for s in set(sets):
            indices = [i for i, v in enumerate(sets) if v == s]
            random.shuffle(indices)
            num_down = random.randint(1, len(indices))
            for i in indices[:num_down]:
                maze[y+1][2*i+1] = 0
                below[i] = True
        # Обновляем множества для следующей строки
        for x in range(width // 2):
            if not below[x]:
                sets[x] = 0
    # Последняя строка: соединяем все множества
    y = height-2
    for x in range(width // 2):
        if sets[x] == 0:
            sets[x] = next_set
            next_set += 1
    for x in range(width // 2 - 1):
        if sets[x] != sets[x+1]:
            maze[y][2*x+2] = 0
            old_set = sets[x+1]
            for i in range(len(sets)):
                if sets[i] == old_set:
                    sets[i] = sets[x]
    for y in range(1, height, 2):
        for x in range(1, width, 2):
            maze[y][x] = 0
    return maze

def generate_maze_aldous_broder(width, height):
    # Алдус-Бродер
    maze = np.ones((height, width), dtype=int)
    y, x = random.randrange(1, height, 2), random.randrange(1, width, 2)
    maze[y][x] = 0
    visited = set([(y, x)])
    total = ((height-1)//2)*((width-1)//2)
    steps = 0  # --- добавлено для контроля зацикливания ---
    max_steps = width * height * 100  # --- ограничение на количество шагов ---
    while len(visited) < total and steps < max_steps:
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        dy, dx = random.choice(dirs)
        ny, nx = y + dy, x + dx
        if 1 <= ny < height-1 and 1 <= nx < width-1:
            if (ny, nx) not in visited:
                maze[y+dy//2][x+dx//2] = 0
                maze[ny][nx] = 0
                visited.add((ny, nx))
            y, x = ny, nx
        steps += 1
    # --- если не удалось сгенерировать, возвращаем None ---
    if len(visited) < total:
        print("Aldous-Broder: не удалось сгенерировать лабиринт за разумное число шагов.")
        return None
    return np.array(maze, dtype=int)

# --- Основная функция генерации с выбором алгоритма ---

def generate_maze(difficulty, force_unsolvable=False, max_attempts=10):
    global maze, entry, exit
    width, height = get_maze_size(difficulty)
    algorithm = algo_combo.get()
    min_path_length = (width + height) // 2

    solvable = False  
    path = []         

    for attempt in range(max_attempts):
        if algorithm == 'Prim (default)' or algorithm == 'Prim':
            maze = generate_maze_prim(width, height)
            algo_name = "prim"
        elif algorithm == 'Kruskal':
            maze = generate_maze_kruskal(width, height)
            algo_name = "kruskal"
        elif algorithm == 'Eller':
            maze = generate_maze_eller(width, height)
            algo_name = "eller"
        elif algorithm == 'Aldous-Бroдер': 
            maze_candidate = generate_maze_aldous_broder(width, height)
            if maze_candidate is None:
                continue  # если генерация не удалась, пробуем ещё раз
            maze = maze_candidate
            algo_name = "aldous_broder"
        else:
            continue

        edges = {
            'top': [(0, x) for x in range(1, width, 2) if maze[1][x] == 0],
            'bottom': [(height - 1, x) for x in range(1, width, 2) if maze[height - 2][x] == 0],
            'left': [(y, 0) for y in range(1, height, 2) if maze[y][1] == 0],
            'right': [(y, width - 1) for y in range(1, height, 2) if maze[y][width - 2] == 0],
        }

        if not any(edges.values()):
            continue

        # Пытаемся найти такие вход и выход, чтобы путь между ними был достаточно длинным
        found = False
        edge_keys = [k for k, v in edges.items() if v]
        for _ in range(20):  # 20 попыток подобрать подходящую пару
            entry_edge = random.choice(edge_keys)
            exit_edge_candidates = [e for e in edge_keys if e != entry_edge]
            if not exit_edge_candidates:
                continue
            exit_edge = random.choice(exit_edge_candidates)
            entry_candidate = random.choice(edges[entry_edge])
            exit_candidate = random.choice(edges[exit_edge])
            maze[entry_candidate] = 0
            maze[exit_candidate] = 0
            solvable, path = is_solvable(maze, entry_candidate, exit_candidate)
            if solvable and len(path) >= min_path_length:
                entry = entry_candidate
                exit = exit_candidate
                found = True
                break
            maze[entry_candidate] = 1
            maze[exit_candidate] = 1
        if not found:
            continue

        solvable, path = is_solvable(maze, entry, exit)
        if force_unsolvable:
            if not solvable:
                break
            maze[path[len(path)//2][0]:path[len(path)//2][0]+2, path[len(path)//2][1]:path[len(path)//2][1]+2] = 1
            if not is_solvable(maze, entry, exit)[0]:
                break
        else:
            if solvable:
                break

    # --- исправление: всегда вычислять path и nn_path_mask, если solvable ---
    nn_path_mask = None
    if solvable and show_nn_path_var.get():
        nn_path_mask = predict_solution(maze)
    if show_path_var.get() and show_nn_path_var.get() and not force_unsolvable and solvable:
        show_maze(path, nn_path_mask)
    elif show_path_var.get() and not force_unsolvable and solvable:
        show_maze(path)
    elif show_nn_path_var.get() and solvable:
        show_maze(nn_path_mask=nn_path_mask)
    else:
        show_maze()

def show_maze(path=None, nn_path_mask=None):
    # --- исправление: убедимся, что maze всегда numpy массив с числовым dtype и двумерный ---
    global maze
    if not isinstance(maze, np.ndarray):
        maze = np.array(maze)
    if maze.dtype == object:
        maze = maze.astype(np.float32)
    # --- добавлено: если maze не двумерный, не отображаем ---
    if maze.ndim != 2:
        print("Ошибка: maze имеет некорректную размерность:", maze.shape)
        plt.close()
        return
    plt.figure(figsize=(10,7))
    plt.imshow(maze, cmap='Greys')
    if path:
        for (y1, x1), (y2, x2) in zip(path[:-1], path[1:]):
            plt.plot([x1, x2], [y1, y2], color='green', linewidth=2, label='BFS путь' if y1 == path[0][0] and x1 == path[0][1] else "")
    if nn_path_mask is not None:
        if nn_path_mask.dtype == object:
            nn_path_mask = nn_path_mask.astype(np.float32)
        if nn_path_mask.ndim == 2:
            plt.imshow(nn_path_mask, cmap='Reds', alpha=0.4)
    # Добавим легенду только если оба пути есть
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