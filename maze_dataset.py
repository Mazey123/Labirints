"""
Модуль для создания PyTorch Dataset, генерирующего лабиринты "на лету".
Это позволяет обучать нейросеть на огромном количестве данных без загрузки их в память.
"""

import random
import logging
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Импортируем логику генерации (предполагается, что maze_generator.py доступен)
# Если функции находятся в том же файле или другом модуле, раскомментируйте и поправьте импорт
# from .maze_generator import generate_maze, MazeAlgorithm, MazeResult 
# Для автономности примера продублируем минимальную логику или импортируем реальную:
try:
    from maze_generator import generate_maze, MazeAlgorithm
except ImportError:
    # Заглушка для примера, если файл maze_generator еще не создан или имя другое
    class MazeAlgorithm(Enum):
        DFS = "dfs"
        PRIM = "prim"
        KRUSKAL = "kruskal"
        ALDOUS_BRODER = "aldous_broder"
        ELLER = "eller"
        RECURSIVE_DIVISION = "recursive_division"

    def generate_maze(width: int, height: int, algorithm: MazeAlgorithm = MazeAlgorithm.DFS) -> Dict[str, Any]:
        """
        Mock implementation for type checking if real module is missing.
        Replace with actual import.
        """
        # Возвращает структуру: {'grid': np.array, 'start': (x,y), 'end': (x,y), 'solution': list}
        grid = np.ones((height, width), dtype=np.uint8) # 1 - стена, 0 - путь
        # Здесь должна быть реальная логика генерации
        # Для теста создадим простой проход
        grid[1:-1, 1:-1] = 0 
        return {
            "grid": grid,
            "start": (1, 1),
            "end": (height-2, width-2),
            "solution": []
        }

logger = logging.getLogger(__name__)


class MazeDataset(Dataset):
    """
    Датасет для генерации лабиринтов на лету.
    
    Атрибуты:
        size_range: Кортеж (min_size, max_size) для случайного выбора размера.
                    Размер должен быть нечетным для большинства алгоритмов.
        algorithms: Список алгоритмов для использования. Если None, используются все.
        seed: Опциональное зерно для воспроизводимости (не рекомендуется для обучения).
        transform: Опциональная трансформация для тензора (аугментация).
    """
    
    def __init__(
        self,
        num_samples: int,  # Виртуальный размер эпохи (сколько раз вызвать __getitem__ за эпоху)
        size_range: Tuple[int, int] = (15, 63),
        algorithms: Optional[List[MazeAlgorithm]] = None,
        seed: Optional[int] = None,
        channel_first: bool = True
    ):
        self.num_samples = num_samples
        self.min_size, self.max_size = size_range
        
        # Убеждаемся, что размеры нечетные (требуется для многих алгоритмов лабиринтов)
        if self.min_size % 2 == 0: self.min_size += 1
        if self.max_size % 2 == 0: self.max_size -= 1
        
        if self.min_size > self.max_size:
            raise ValueError(f"Некорректный диапазон размеров: {size_range}")

        self.algorithms = algorithms if algorithms else list(MazeAlgorithm)
        self.channel_first = channel_first
        
        # Локальный генератор случайных чисел для каждого воркера
        # Это критически важно для корректной работы с num_workers > 0
        self.worker_rng = random.Random()
        
        if seed is not None:
            logger.warning("Установка seed в датасете может снизить разнообразие данных при обучении.")
            self.worker_rng.seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __worker_init_fn(self, worker_id: int):
        """
        Инициализируется каждым воркером DataLoader.
        Создает уникальный seed для каждого воркера на основе base seed и ID воркера.
        """
        # Получаем базовый seed из главного процесса (если есть) или используем время
        base_seed = torch.initial_seed() % (2**32)
        self.worker_rng.seed(base_seed + worker_id)
        logger.debug(f"Worker {worker_id} initialized with seed {base_seed + worker_id}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Генерирует один лабиринт.
        
        Returns:
            Dict с тензорами:
                'image': Входное изображение (стены/пути).
                'start_point': Координаты старта.
                'end_point': Координаты финиша.
                'solution_mask': Маска правильного пути (для обучения сегментации).
        """
        # 1. Случайный выбор параметров
        # Выбираем нечетный размер из диапазона
        step = 2
        current_size = self.worker_rng.randrange(self.min_size, self.max_size + 1, step)
        
        width = current_size
        height = current_size # Можно сделать прямоугольные, если изменить логику
        
        algorithm = self.worker_rng.choice(self.algorithms)
        
        # 2. Генерация лабиринта
        try:
            result = generate_maze(width, height, algorithm)
            grid = result['grid'] # Ожидаем numpy array (H, W)
            start = result['start']
            end = result['end']
            solution_path = result.get('solution', [])
        except Exception as e:
            logger.error(f"Ошибка генерации лабиринта (alg={algorithm}, size={width}): {e}")
            # Фолбэк: возвращаем пустой или простой лабиринт, чтобы обучение не упало
            grid = np.ones((height, width), dtype=np.uint8)
            start = (1, 1)
            end = (height-2, width-2)
            solution_path = []

        # 3. Подготовка данных для нейросети
        
        # Нормализация: 0 -> путь, 1 -> стена. 
        # Для CNN часто удобнее: 0.0 (путь) и 1.0 (стена) или наоборот.
        # Приводим к float32
        image_tensor = torch.from_numpy(grid).float()
        
        # Создаем маску решения (сегментация)
        solution_mask = np.zeros_like(grid, dtype=np.float32)
        for y, x in solution_path:
            if 0 <= y < height and 0 <= x < width:
                solution_mask[y, x] = 1.0
        solution_tensor = torch.from_numpy(solution_mask)

        # Добавляем каналы, если нужно (C, H, W)
        if self.channel_first:
            image_tensor = image_tensor.unsqueeze(0)      # (1, H, W)
            solution_tensor = solution_tensor.unsqueeze(0)# (1, H, W)
            
            # Опционально: сделать 3 канала для совместимости с предобученными сетями (ResNet и т.д.)
            # image_tensor = image_tensor.repeat(3, 1, 1)

        # Координаты старта и конца можно вернуть как тензор [y, x] или просто сохранить в метаданных
        start_tensor = torch.tensor([start[0], start[1]], dtype=torch.float32)
        end_tensor = torch.tensor([end[0], end[1]], dtype=torch.float32)

        return {
            'image': image_tensor,
            'target': solution_tensor, # Целевая маска пути
            'start': start_tensor,
            'end': end_tensor,
            'metadata': {
                'size': (height, width),
                'algorithm': algorithm.name
            }
        }


def get_maze_dataloader(
    samples_per_epoch: int = 10000,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    Фабричная функция для создания DataLoader с правильной инициализацией воркеров.
    
    Args:
        samples_per_epoch: Количество лабиринтов за одну эпоху обучения.
        batch_size: Размер батча.
        num_workers: Количество потоков для загрузки данных.
        **dataset_kwargs: Аргументы для MazeDataset.
    """
    
    dataset = MazeDataset(num_samples=samples_per_epoch, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Шаффл важен, так как данные генерируются случайно
        num_workers=num_workers,
        pin_memory=True, # Ускоряет передачу на GPU
        worker_init_fn=dataset._MazeDataset__worker_init_fn, # Доступ к приватному методу инициализации
        persistent_workers=True if num_workers > 0 else False # Не перезапускать воркеров каждую эпоху
    )
    
    return dataloader


if __name__ == "__main__":
    # Пример использования и тестирования
    logging.basicConfig(level=logging.INFO)
    
    print("Тестирование MazeDataset...")
    
    # Создаем датасет на 100 элементов для теста
    ds = MazeDataset(num_samples=100, size_range=(15, 31))
    
    # Берем один элемент
    sample = ds[0]
    print(f"Размер изображения: {sample['image'].shape}")
    print(f"Размер маски: {sample['target'].shape}")
    print(f"Алгоритм: {sample['metadata']['algorithm']}")
    print(f"Старт: {sample['start']}, Финиш: {sample['end']}")
    
    # Тестируем DataLoader
    print("\nТестирование DataLoader...")
    loader = get_maze_dataloader(samples_per_epoch=50, batch_size=4, num_workers=2)
    
    for i, batch in enumerate(loader):
        if i >= 2: break # Покажем только 2 батча
        
        print(f"Batch {i}:")
        print(f"  Images shape: {batch['image'].shape}") # (B, C, H, W) - но H, W могут отличаться!
        # ВАЖНО: Если размеры лабиринтов разные, стандартный DataLoader выдаст ошибку коллации.
        # Нужно использовать custom collate_fn или фиксированный размер.
        # Ниже пример обработки разных размеров.
        break

    print("\nВнимание: Если размеры лабиринтов разные, нужен custom collate_fn или ресайз.")
