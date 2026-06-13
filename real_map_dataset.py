import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from real_map_config import RealMapConfig

# Именованные функции для трансформаций (для совместимости с multiprocessing)
def invert_image(x):
    return 1 - x

def binarize_mask(x):
    return (x > 0.5).float()

class RealMapDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.img_files) == len(self.mask_files), f"Mismatch: {len(self.img_files)} images, {len(self.mask_files)} masks"

        # Трансформация для изображения
        self.img_transform = T.Compose([
            T.Resize((RealMapConfig.IMG_SIZE, RealMapConfig.IMG_SIZE)),
            T.ToTensor(),
            T.Lambda(invert_image)
        ])

        # Трансформация для маски
        self.mask_transform = T.Compose([
            T.Resize((RealMapConfig.IMG_SIZE, RealMapConfig.IMG_SIZE), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            T.Lambda(binarize_mask)
        ])

        if self.augment:
            self.geometric = T.Compose([
                T.RandomHorizontalFlip(p=RealMapConfig.HFLIP_PROB),
                T.RandomVerticalFlip(p=RealMapConfig.VFLIP_PROB),
                T.RandomRotation(degrees=RealMapConfig.ROTATION_DEG, fill=0)
            ])
        else:
            self.geometric = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.geometric is not None:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            img = self.geometric(img)
            torch.manual_seed(seed)
            mask = self.geometric(mask)

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask

# ---------- ЭТА ФУНКЦИЯ ВАМ НУЖНА ----------
def get_real_map_loaders():
    """Создаёт DataLoader для обучения и валидации."""
    train_dataset = RealMapDataset(RealMapConfig.TRAIN_IMG_DIR, RealMapConfig.TRAIN_MASK_DIR, augment=True)
    val_dataset = RealMapDataset(RealMapConfig.VAL_IMG_DIR, RealMapConfig.VAL_MASK_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=RealMapConfig.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=RealMapConfig.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader