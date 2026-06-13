import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from real_map_config import RealMapConfig
from real_map_dataset import get_real_map_loaders
from train_maze_model import UNet

# Временный класс для совместимости со старым чекпоинтом
class TrainingConfig:
    pass

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    for images, masks in tqdm(loader, desc="Validating"):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        intersection = (preds * masks).sum()
        union = preds.sum() + masks.sum() - intersection
        iou = (intersection / (union + 1e-6)).item()
        total_iou += iou
    return total_loss / len(loader), total_iou / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(RealMapConfig.SAVE_DIR, exist_ok=True)

    train_loader, val_loader = get_real_map_loaders()
    if len(train_loader.dataset) == 0:
        print("No training data. Run generate_real_map_dataset.py first.")
        return

    if not os.path.exists(RealMapConfig.PRETRAINED_WEIGHTS):
        print(f"Pretrained weights not found at {RealMapConfig.PRETRAINED_WEIGHTS}")
        return

    # Загружаем модель
    model = UNet(in_channels=RealMapConfig.IN_CHANNELS, out_channels=1).to(device)

    # Универсальная загрузка весов (без необходимости в TrainingConfig)
    checkpoint = torch.load(RealMapConfig.PRETRAINED_WEIGHTS, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Если чекпоинт — просто словарь весов (state_dict)
        model.load_state_dict(checkpoint)

    print("Pretrained weights loaded.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=RealMapConfig.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_iou = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

    for epoch in range(RealMapConfig.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f"Epoch {epoch+1}/{RealMapConfig.EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, os.path.join(RealMapConfig.SAVE_DIR, "real_map_best_iou.pth"))
            print(f"  Best model saved (IoU={val_iou:.4f})")

    # Финальная модель
    torch.save(model.state_dict(), os.path.join(RealMapConfig.SAVE_DIR, "real_map_final.pth"))
    print(f"Training finished. Best IoU: {best_iou:.4f}")

    # График
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['val_iou'], label='Val IoU', color='green')
    plt.legend()
    plt.title('IoU')
    plt.savefig(os.path.join(RealMapConfig.SAVE_DIR, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    main()