"""
Скрипт для визуальной оценки обученной модели лабиринтов.
Показывает:
- Входной лабиринт
- Предсказание модели (маска пути)
- Правильный путь (ground truth)
- Разницу между предсказанием и истиной
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train_maze_model import (
    MazeAlgorithm,
    generate_maze_with_solution,
    UNet,
    TrainingConfig
)


def load_model(model_path: str, device: str = 'cpu') -> UNet:
    """Загрузка обученной модели."""
    print(f"Загрузка модели из {model_path}...")

    # Создаём модель (архитектура из train_maze_model.py)
    model = UNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256]
    ).to(device)

    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Эпоха: {checkpoint.get('epoch', 'N/A')}")
        print(f"Val F1: {checkpoint.get('val_f1', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Модель загружена успешно!")
    return model


def visualize_prediction(
    model: UNet,
    maze_data: dict,
    device: str = 'cpu',
    threshold: float = 0.5
):
    """Визуализация предсказания модели для одного лабиринта."""

    maze = maze_data['maze']  # 0=проход, 1=стена (как в оригинале)
    solution = maze_data['solution']

    # Инвертируем для подачи в модель (как при обучении: проходы=1, стены=0)
    maze_input = 1 - maze

    # Подготовка тензора
    maze_tensor = torch.tensor(maze_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    maze_tensor = maze_tensor.to(device)

    # Предсказание
    with torch.no_grad():
        prediction = model(maze_tensor)
        prediction = F.sigmoid(prediction).squeeze().cpu().numpy()

    # Бинаризация
    pred_binary = (prediction > threshold).astype(np.float32)

    # Metrics
    intersection = np.logical_and(pred_binary, solution).sum()
    union = np.logical_or(pred_binary, solution).sum()
    iou = intersection / union if union > 0 else 0

    precision = intersection / pred_binary.sum() if pred_binary.sum() > 0 else 0
    recall = intersection / solution.sum() if solution.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'maze': maze,
        'solution': solution,
        'prediction': prediction,
        'prediction_binary': pred_binary,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_results(results: list, save_path: str = "model_evaluation.png"):
    """Построение графиков с результатами."""
    n_samples = len(results)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for i, res in enumerate(results):
        row = axes[i] if n_samples > 1 else axes

        # 1. Лабиринт
        ax = row[0]
        ax.imshow(res['maze'], cmap='binary')
        ax.set_title(f"Лабиринт {i+1}\n(стены=чёрные, проходы=белые)")
        ax.axis('off')

        # 2. Правильный путь
        ax = row[1]
        ax.imshow(res['solution'], cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f"Правильный путь\n(Ground Truth)")
        ax.axis('off')

        # 3. Предсказание модели
        ax = row[2]
        im = ax.imshow(res['prediction'], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"Предсказание модели\nF1={res['f1']:.3f}, IoU={res['iou']:.3f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 4. Разница
        ax = row[3]
        diff = np.abs(res['prediction'] - res['solution'])
        im = ax.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f"Ошибка\n(красный=ошибка, зелёный=точно)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nРезультаты сохранены в {save_path}")
    plt.show()


def main():
    # Конфигурация
    MODEL_PATH = "maze_model_best_f1.pth"  # Или путь к вашей модели
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_SAMPLES = 5  # Количество тестовых лабиринтов
    SIZE_RANGE = (33, 77)
    THRESHOLD = 0.5

    print(f"Используемое устройство: {DEVICE}")

    # Проверка наличия модели
    if not os.path.exists(MODEL_PATH):
        # Пробуем в подпапке
        alt_path = os.path.join("Maze models", "maze_model_best_f1_first.pth")
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
        else:
            print(f"ERROR: Модель не найдена: {MODEL_PATH}")
            print("Доступные модели:")
            for root, dirs, files in os.walk("."):
                for f in files:
                    if f.endswith('.pth'):
                        print(f"  - {os.path.join(root, f)}")
            return

    # Загрузка модели
    model = load_model(MODEL_PATH, DEVICE)

    # Генерация тестовых лабиринтов
    print(f"\nГенерация {N_SAMPLES} тестовых лабиринтов...")
    results = []

    for i in range(N_SAMPLES):
        size = random.randrange(SIZE_RANGE[0], SIZE_RANGE[1], 2)
        algorithm = random.choice(list(MazeAlgorithm))

        # Генерация
        max_attempts = 10
        for attempt in range(max_attempts):
            data = generate_maze_with_solution(size, algorithm)
            if data is not None:
                break

        if data is None:
            print(f"  ⚠ Не удалось сгенерировать лабиринт #{i+1}")
            continue

        # Предсказание
        res = visualize_prediction(model, data, DEVICE, THRESHOLD)
        res['size'] = size
        res['algorithm'] = algorithm.name

        results.append(res)

        print(f"  ✓ Лабиринт #{i+1}: размер={size}, алгоритм={algorithm.name}, "
              f"F1={res['f1']:.3f}, IoU={res['iou']:.3f}")

    if not results:
        print("\nERROR: Не удалось сгенерировать ни одного тестового лабиринта")
        return

    # Статистика
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    print(f"\n{'='*50}")
    print(f"Средний F1: {avg_f1:.3f}")
    print(f"Средний IoU: {avg_iou:.3f}")
    print(f"{'='*50}")

    # Визуализация
    plot_results(results, "model_evaluation.png")


if __name__ == "__main__":
    main()