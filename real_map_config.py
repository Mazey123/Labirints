import os
from dataclasses import dataclass

@dataclass
class RealMapConfig:
    # Пути к данным (будут созданы скриптом generate_real_map_dataset.py)
    DATA_ROOT = "data/real_maps_thick"
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train", "masks")
    VAL_IMG_DIR = os.path.join(DATA_ROOT, "val", "images")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "val", "masks")

    # Модель
    PRETRAINED_WEIGHTS = "maze_model_best_f1.pth"   # веса, обученные на синтетике
    SAVE_DIR = "models_real_maps"

    # Параметры изображения
    IMG_SIZE = 526
    IN_CHANNELS = 1

    # Обучение
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    # DEVICE будет определён в train/evaluate скриптах

    # Аугментация
    ROTATION_DEG = 15
    HFLIP_PROB = 0.5
    VFLIP_PROB = 0.5