"""
Усовершенствованный тренировочный скрипт для U-Net модели лабиринтов.

Возможности:
- Генерация лабиринтов "на лету" (без сохранения на диск)
- Поддержка разных размеров лабиринтов
- Валидация и Early Stopping
- Аугментация данных
- Динамический расчет pos_weight
- Learning Rate scheduler
- Сохранение лучших чекпоинтов
- Поддержка перехода к реальным картам

Рекомендации:
- Batch Size: 32-64 (зависит от GPU памяти)
- Эпохи: 50-100 с early stopping
- Learning Rate: 1e-4 начальный, с уменьшением
- Количество данных: 200,000 - 500,000 лабиринтов минимум
"""

import os
import random
import time
import logging
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

@dataclass
class TrainingConfig:
    """Конфигурация обучения."""
    # Параметры данных
    samples_per_epoch: int = 50000  # Сколько лабиринтов за эпоху
    validation_samples: int = 5000   # Сколько лабиринтов для валидации
    size_range: Tuple[int, int] = (33, 77)  # Диапазон размеров (нечетные)
    
    # Параметры обучения
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    
    # Early Stopping
    patience: int = 10  # Эпох без улучшения
    min_delta: float = 0.001  # Минимальное улучшение для F1
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'plateau', 'cosine', 'none'
    T_max: int = 50  # Для cosine scheduler
    
    # Сохранение
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "maze_model_best_f1.pth"
    
    # Устройство
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# ГЕНЕРАТОР ЛАБИРИНТОВ (встроенный)
# ============================================================================

class MazeAlgorithm(Enum):
    PRIM = "prim"
    KRUSKAL = "kruskal"
    ELLER = "eller"
    ALDOUS_BRODER = "aldous_broder"


def generate_maze_prim(width: int, height: int) -> np.ndarray:
    """Генерация лабиринта алгоритмом Прима."""
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
    """Генерация лабиринта алгоритмом Краскала."""
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
    """Генерация лабиринта алгоритмом Эллера."""
    if width % 2 == 0 or height % 2 == 0:
        return None
    
    maze = np.ones((height, width), dtype=np.uint8)
    sets = [0] * (width // 2)
    next_set = 1
    
    for y in range(1, height - 2, 2):
        for x in range(width // 2):
            if sets[x] == 0:
                sets[x] = next_set
                next_set += 1
        
        for x in range(width // 2 - 1):
            if sets[x] != sets[x + 1] and random.choice([True, False]):
                maze[y, 2 * x + 2] = 0
                old_set = sets[x + 1]
                for i in range(len(sets)):
                    if sets[i] == old_set:
                        sets[i] = sets[x]
        
        below = [False] * (width // 2)
        unique_sets = set(sets)
        for s in unique_sets:
            indices = [i for i, v in enumerate(sets) if v == s]
            random.shuffle(indices)
            num_down = random.randint(1, len(indices))
            for i in indices[:num_down]:
                maze[y + 1, 2 * i + 1] = 0
                below[i] = True
        
        for x in range(width // 2):
            if not below[x]:
                sets[x] = 0
    
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
    
    for y in range(1, height, 2):
        for x in range(1, width, 2):
            maze[y, x] = 0
    
    return maze


def generate_maze_aldous_broder(width: int, height: int, max_steps_factor: int = 100) -> Optional[np.ndarray]:
    """Генерация лабиринта алгоритмом Aldous-Broder."""
    maze = np.ones((height, width), dtype=np.uint8)
    
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
        return None
    
    return maze


def find_solution_path(maze: np.ndarray) -> Tuple[bool, List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    """
    Находит путь через лабиринт и возвращает вход/выход.
    Returns: (is_solvable, path, entry_point, exit_point)
    """
    from collections import deque
    
    height, width = maze.shape
    
    # Находим все возможные входы/выходы на краях
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
        return False, [], None, None
    
    # Ищем путь между всеми парами
    for entry in entries:
        for exit_point in exits:
            # BFS
            visited = {entry: None}
            queue = deque([entry])
            
            while queue:
                y, x = queue.popleft()
                if (y, x) == exit_point:
                    # Восстанавливаем путь
                    path = []
                    current = (y, x)
                    while current is not None:
                        path.append(current)
                        current = visited[current]
                    return True, path[::-1], entry, exit_point
                
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and 
                        maze[ny, nx] == 0 and (ny, nx) not in visited):
                        visited[(ny, nx)] = (y, x)
                        queue.append((ny, nx))
    
    return False, [], None, None


def generate_maze_with_solution(
    size: int,
    algorithm: MazeAlgorithm = MazeAlgorithm.PRIM
) -> Optional[Dict[str, Any]]:
    """
    Генерирует лабиринт с решением.
    Returns dict с maze, solution_mask, entry, exit
    """
    width = height = size if size % 2 == 1 else size + 1
    
    # Генерация базового лабиринта
    if algorithm == MazeAlgorithm.PRIM:
        maze = generate_maze_prim(width, height)
    elif algorithm == MazeAlgorithm.KRUSKAL:
        maze = generate_maze_kruskal(width, height)
    elif algorithm == MazeAlgorithm.ELLER:
        maze = generate_maze_eller(width, height)
        if maze is None:
            return None
    elif algorithm == MazeAlgorithm.ALDOUS_BRODER:
        maze = generate_maze_aldous_broder(width, height)
        if maze is None:
            return None
    else:
        maze = generate_maze_prim(width, height)
    
    # Добавляем входы/выходы
    edges = {
        'top': [(0, x) for x in range(1, width, 2) if maze[1, x] == 0],
        'bottom': [(height - 1, x) for x in range(1, width, 2) if maze[height - 2, x] == 0],
        'left': [(y, 0) for y in range(1, height, 2) if maze[y, 1] == 0],
        'right': [(y, width - 1) for y in range(1, height, 2) if maze[y, width - 2] == 0],
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
    
    # Находим решение
    is_solvable, path, entry, exit_point = find_solution_path(maze)
    
    if not is_solvable or not path:
        return None
    
    # Создаем маску решения
    solution_mask = np.zeros_like(maze, dtype=np.float32)
    for y, x in path:
        if 0 <= y < height and 0 <= x < width:
            solution_mask[y, x] = 1.0
    
    return {
        'maze': maze.astype(np.float32),
        'solution': solution_mask,
        'entry': entry,
        'exit': exit_point,
        'algorithm': algorithm.name,
        'size': (height, width)
    }


# ============================================================================
# DATASET С ГЕНЕРАЦИЕЙ "НА ЛЕТУ"
# ============================================================================

class MazeDatasetOnTheFly(Dataset):
    """
    Датасет с генерацией лабиринтов на лету.
    Не требует сохранения файлов на диск.
    """
    
    def __init__(
        self,
        num_samples: int,
        size_range: Tuple[int, int] = (33, 77),
        algorithms: Optional[List[MazeAlgorithm]] = None,
        augment: bool = True,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.min_size, self.max_size = size_range
        
        # Убеждаемся, что размеры нечетные
        if self.min_size % 2 == 0:
            self.min_size += 1
        if self.max_size % 2 == 0:
            self.max_size -= 1
        
        if self.min_size > self.max_size:
            raise ValueError(f"Некорректный диапазон размеров: {size_range}")
        
        self.algorithms = algorithms if algorithms else list(MazeAlgorithm)
        self.augment = augment
        self.worker_rng = random.Random()
        self.base_seed = seed if seed is not None else 42
        
        if seed is not None:
            self.worker_rng.seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def _worker_init_fn(self, worker_id: int):
        """Инициализация RNG для каждого worker."""
        base_seed = torch.initial_seed() % (2**32)
        self.worker_rng.seed(base_seed + worker_id + self.base_seed)
    
    def _augment(self, maze: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Аугментация данных: вращения и отражения."""
        k = self.worker_rng.randint(0, 3)  # Количество поворотов на 90 градусов
        maze = np.rot90(maze, k=k).copy()
        solution = np.rot90(solution, k=k).copy()
        
        if self.worker_rng.random() > 0.5:
            maze = np.fliplr(maze).copy()
            solution = np.fliplr(solution).copy()
        
        if self.worker_rng.random() > 0.5:
            maze = np.flipud(maze).copy()
            solution = np.flipud(solution).copy()
        
        return maze, solution
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Случайный выбор параметров
        step = 2
        size = self.worker_rng.randrange(self.min_size, self.max_size + 1, step)
        algorithm = self.worker_rng.choice(self.algorithms)
        
        # Генерация лабиринта
        max_attempts = 5
        result = None
        for attempt in range(max_attempts):
            result = generate_maze_with_solution(size, algorithm)
            if result is not None:
                break
        
        if result is None:
            # Fallback: простой лабиринт
            logger.warning(f"Не удалось сгенерировать лабиринт, используем fallback")
            maze = np.ones((size, size), dtype=np.float32)
            maze[1:-1:2, 1:-1:2] = 0
            solution = np.zeros_like(maze, dtype=np.float32)
            solution[size//2, :] = 1.0
        else:
            maze = result['maze']
            solution = result['solution']
        
        # Аугментация
        if self.augment:
            maze, solution = self._augment(maze, solution)
        
        # Инверсия: проходы=1, стены=0 (как в оригинальном коде)
        maze = 1 - maze
        
        # Конвертация в тензоры
        maze_tensor = torch.tensor(maze, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        solution_tensor = torch.tensor(solution, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        
        return {
            'image': maze_tensor,
            'target': solution_tensor,
            'metadata': {
                'size': (maze.shape[0], maze.shape[1]),
                'algorithm': algorithm.name
            }
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function для обработки разных размеров лабиринтов.
    Padding до максимального размера в батче.
    """
    # Получаем размеры всех изображений в батче
    batch_size = len(batch)
    
    # Находим максимальные H и W
    max_h = 0
    max_w = 0
    for item in batch:
        img = item['image']
        if len(img.shape) == 3:
            h, w = img.shape[1], img.shape[2]
        else:
            h, w = img.shape[2], img.shape[3]
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    
    # Делаем размеры нечетными (требуется для U-Net с pooling)
    if max_h % 2 == 0:
        max_h += 1
    if max_w % 2 == 0:
        max_w += 1
    
    # Создаем тензоры с padding
    images = torch.zeros(batch_size, 1, max_h, max_w, dtype=torch.float32)
    targets = torch.zeros(batch_size, 1, max_h, max_w, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        img = item['image']
        tgt = item['target']
        
        # Сжимаем до (H, W) если есть канал
        if len(img.shape) == 3:
            img_2d = img.squeeze(0)
            tgt_2d = tgt.squeeze(0)
        else:
            img_2d = img
            tgt_2d = tgt
        
        h, w = img_2d.shape[0], img_2d.shape[1]
        
        # Копируем в верхний левый угол
        images[i, 0, :h, :w] = img_2d
        targets[i, 0, :h, :w] = tgt_2d
    
    return {
        'image': images,
        'target': targets
    }


# ============================================================================
# U-NET МОДЕЛЬ
# ============================================================================

class UNet(nn.Module):
    """U-Net архитектура для сегментации лабиринтов."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: List[int] = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Down part
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
        )
        
        # Up part
        up_in_channels = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(up_in_channels, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            up_in_channels = feature
        
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)
        
        return self.final_conv(x)


# ============================================================================
# ТРЕНИРОВКА
# ============================================================================

class EarlyStopping:
    """Early stopping для остановки обучения при отсутствии улучшений."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def calculate_pos_weight(loader: DataLoader, device: torch.device, num_batches: int = 100) -> torch.Tensor:
    """Динамический расчет pos_weight на основе выборки данных.
    Использует отдельный dataset для избежания проблем с collate_fn."""
    total_pos = 0
    total_neg = 0
    
    # Создаем новый dataset с фиксированным размером для стабильности
    from train_maze_model import MazeDatasetOnTheFly
    temp_dataset = MazeDatasetOnTheFly(
        num_samples=num_batches * loader.batch_size,
        size_range=(33, 33),  # Фиксированный размер чтобы избежать проблем с padding
        augment=False,
        seed=123
    )
    
    temp_loader = DataLoader(
        temp_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    try:
        for i, batch in enumerate(temp_loader):
            if i >= num_batches:
                break
            
            target = batch['target']
            total_pos += (target == 1).sum().item()
            total_neg += (target == 0).sum().item()
    except Exception as e:
        logger.warning(f"Ошибка при расчете pos_weight: {e}")
        logger.warning("Используем значение по умолчанию 10.0")
        return torch.tensor([10.0], dtype=torch.float32, device=device)
    
    if total_pos == 0:
        logger.warning("Не найдено положительных пикселей! Используем вес 1.0")
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    
    pos_weight = torch.tensor([total_neg / total_pos], dtype=torch.float32, device=device)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)
    logger.info(f"Найдено {total_pos/(total_pos+total_neg)*100:.2f}% положительных пикселей")
    logger.info(f"Расчетный pos_weight: {pos_weight.item():.2f}")
    return pos_weight


def train(config: TrainingConfig = None, continue_from_checkpoint: bool = False):
    """
    Основная функция обучения.
    
    Args:
        config: Конфигурация обучения
        continue_from_checkpoint: Продолжить с чекпоинта
    """
    if config is None:
        config = TrainingConfig()
    
    logger.info(f"Используемое устройство: {config.device}")
    if config.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Создание датасетов
    logger.info("Создание датасетов...")
    train_dataset = MazeDatasetOnTheFly(
        num_samples=config.samples_per_epoch,
        size_range=config.size_range,
        augment=True
    )
    
    val_dataset = MazeDatasetOnTheFly(
        num_samples=config.validation_samples,
        size_range=config.size_range,
        augment=False,
        seed=42  # Фиксированный seed для воспроизводимости валидации
    )
    
    # DataLoaders
    logger.info("Создание DataLoader...")
    
    # Для стабильности используем num_workers=0 если возникают проблемы
    # Или уменьшаем до 2 если много памяти
    safe_num_workers = min(config.num_workers, 2) if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=safe_num_workers,  # Безопасное значение
        pin_memory=config.pin_memory if safe_num_workers > 0 else False,
        collate_fn=collate_fn,
        persistent_workers=False,  # Отключаем для стабильности
        worker_init_fn=train_dataset._worker_init_fn if hasattr(train_dataset, '_worker_init_fn') else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=safe_num_workers,  # Безопасное значение
        pin_memory=config.pin_memory if safe_num_workers > 0 else False,
        collate_fn=collate_fn,
        persistent_workers=False,  # Отключаем для стабильности
        worker_init_fn=val_dataset._worker_init_fn if hasattr(val_dataset, '_worker_init_fn') else None
    )
    
    # Модель
    model = UNet(in_channels=1, out_channels=1).to(config.device)
    
    # Оптимизатор
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Loss function с динамическим pos_weight
    logger.info("Расчет pos_weight...")
    pos_weight = calculate_pos_weight(train_loader, config.device)
    logger.info(f"pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Scheduler
    if config.scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif config.scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_max, T_mult=2)
    else:
        scheduler = None
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, mode='max')
    
    # Загрузка чекпоинта
    best_f1 = 0.0
    start_epoch = 0
    
    if continue_from_checkpoint and os.path.exists(config.best_model_path):
        checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_f1 = checkpoint.get('best_f1', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Загружен чекпоинт из эпохи {start_epoch-1}, best_f1={best_f1:.4f}")
    
    # Scaler для mixed precision
    scaler = GradScaler(device=config.device if config.device == 'cuda' else 'cpu')
    
    # Метрики
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    logger.info(f"Начало обучения с эпохи {start_epoch}...")
    logger.info(f"Всего эпох: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Эпоха {epoch+1}/{config.num_epochs}")
        logger.info(f"{'='*60}")
        
        # === TRAINING ===
        model.train()
        total_train_loss = 0.0
        train_f1_sum = 0.0
        n_train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(config.device)
            targets = batch['target'].to(config.device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if config.device == 'cuda' else 'cpu'):
                outputs = model(images)
                
                # Обрезаем output до размера target (если есть padding)
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = outputs[:, :, :targets.shape[2], :targets.shape[3]]
                
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            # Расчет метрик
            preds = (torch.sigmoid(outputs) > 0.5).detach().cpu().numpy().astype(int).flatten()
            trues = targets.detach().cpu().numpy().astype(int).flatten()
            
            train_f1_sum += f1_score(trues, preds, zero_division=0)
            n_train_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Train batch {batch_idx+1}/{len(train_loader)} | loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / n_train_batches
        avg_train_f1 = train_f1_sum / n_train_batches
        
        # === VALIDATION ===
        model.eval()
        total_val_loss = 0.0
        val_f1_sum = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(config.device)
                targets = batch['target'].to(config.device)
                
                outputs = model(images)
                
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = outputs[:, :, :targets.shape[2], :targets.shape[3]]
                
                loss = criterion(outputs, targets)
                
                total_val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).detach().cpu().numpy().astype(int).flatten()
                trues = targets.detach().cpu().numpy().astype(int).flatten()
                
                val_f1_sum += f1_score(trues, preds, zero_division=0)
                n_val_batches += 1
        
        avg_val_loss = total_val_loss / n_val_batches
        avg_val_f1 = val_f1_sum / n_val_batches
        
        # Scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if config.scheduler_type == 'plateau':
                scheduler.step(avg_val_f1)
            elif config.scheduler_type == 'cosine':
                scheduler.step(epoch)
                current_lr = scheduler.get_last_lr()[0]
        
        # Логирование
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Train F1: {avg_train_f1:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f} | Val F1: {avg_val_f1:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Сохранение истории
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_f1'].append(avg_train_f1)
        history['val_f1'].append(avg_val_f1)
        history['learning_rates'].append(current_lr)
        
        # Сохранение лучшей модели
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config
            }
            torch.save(checkpoint, config.best_model_path)
            logger.info(f"✨ Сохранена новая лучшая модель! Val F1: {best_f1:.4f}")
        
        # Early Stopping
        if early_stopping(avg_val_f1):
            logger.info(f"⏹️  Early Stopping на эпохе {epoch+1}")
            break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Обучение завершено!")
    logger.info(f"Лучший Val F1: {best_f1:.4f}")
    logger.info(f"{'='*60}")
    
    # Визуализация
    plot_history(history, config.checkpoint_dir)
    
    return model, history


def plot_history(history: Dict[str, List[float]], save_dir: str = "."):
    """Визуализация истории обучения."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(history['train_f1'], label='Train F1', linewidth=2)
    axes[0, 1].plot(history['val_f1'], label='Val F1', linewidth=2)
    axes[0, 1].set_title('F1 Score', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    if history['learning_rates']:
        axes[1, 0].plot(history['learning_rates'], linewidth=2, color='red')
        axes[1, 0].set_title('Learning Rate', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Train vs Val F1 Difference
    diff = [abs(t - v) for t, v in zip(history['train_f1'], history['val_f1'])]
    axes[1, 1].plot(diff, linewidth=2, color='purple')
    axes[1, 1].set_title('Train-Val F1 Gap', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    logger.info(f"Графики сохранены в {save_dir}/training_history.png")


# ============================================================================
# ДЛЯ ПЕРЕХОДА К РЕАЛЬНЫМ КАРТАМ
# ============================================================================

class RealMapDataset(Dataset):
    """
    Датасет для реальных карт (городские карты, спутниковые снимки и т.д.)
    
    Пример использования:
    - Вход: карта города (здания=1, дороги=0)
    - Выход: оптимальный маршрут между точками
    
    Можно адаптировать под:
    - Навигационные карты
    - Карты помещений
    - Игровые карты
    """
    
    def __init__(
        self,
        data_dir: str,
        image_suffix: str = "_input.npy",
        target_suffix: str = "_route.npy",
        augment: bool = True
    ):
        self.data_dir = data_dir
        self.image_suffix = image_suffix
        self.target_suffix = target_suffix
        self.augment = augment
        
        self.files = [
            f.replace(image_suffix, '')
            for f in os.listdir(data_dir)
            if f.endswith(image_suffix)
        ]
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_base = self.files[idx]
        
        image = np.load(os.path.join(self.data_dir, file_base + self.image_suffix))
        target = np.load(os.path.join(self.data_dir, file_base + self.target_suffix))
        
        # Аугментация (аналогично MazeDatasetOnTheFly)
        if self.augment and random.random() > 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k).copy()
            target = np.rot90(target, k=k).copy()
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
        return {'image': image, 'target': target}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training script for maze U-Net")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--samples-per-epoch', type=int, default=50000, help='Samples per epoch')
    parser.add_argument('--size-min', type=int, default=33, help='Min maze size')
    parser.add_argument('--size-max', type=int, default=77, help='Max maze size')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        samples_per_epoch=args.samples_per_epoch,
        size_range=(args.size_min, args.size_max),
        patience=args.patience
    )
    
    logger.info("="*60)
    logger.info("Усовершенствованное обучение U-Net для лабиринтов")
    logger.info("="*60)
    logger.info(f"Конфигурация:")
    logger.info(f"  - Эпохи: {config.num_epochs}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Samples per epoch: {config.samples_per_epoch}")
    logger.info(f"  - Размер лабиринтов: {config.size_range}")
    logger.info(f"  - Early stopping patience: {config.patience}")
    logger.info("="*60)
    
    model, history = train(config, continue_from_checkpoint=args.resume)
    
    logger.info("\n✅ Обучение завершено успешно!")
    logger.info(f"📁 Лучшая модель сохранена в: {config.best_model_path}")
