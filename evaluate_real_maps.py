import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from real_map_config import RealMapConfig
from train_maze_model import UNet   

def load_model(path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    chk = torch.load(path, map_location=device, weights_only=False)
    if isinstance(chk, dict) and 'model_state_dict' in chk:
        model.load_state_dict(chk['model_state_dict'])
    elif isinstance(chk, dict) and 'state_dict' in chk:
        model.load_state_dict(chk['state_dict'])
    else:
        model.load_state_dict(chk)
    model.eval()
    return model

def predict(image_path, model, device):
    img = Image.open(image_path).convert('L')
    # Трансформация как в датасете: resize -> toTensor -> инверсия (дороги=1)
    transform = T.Compose([
        T.Resize((RealMapConfig.IMG_SIZE, RealMapConfig.IMG_SIZE)),
        T.ToTensor(),
        T.Lambda(lambda x: 1 - x)
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).cpu().squeeze().numpy()
    return prob

def visualize(image_path, prob, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    orig = Image.open(image_path).convert('L')
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title('Input (roads=black, walls=white)')
    axes[0].axis('off')
    im = axes[1].imshow(prob, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Prediction probability')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    binary = (prob > 0.5).astype(np.uint8)
    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title('Binary prediction (roads=white)')
    axes[2].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Пробуем загрузить дообученную модель, иначе синтетическую
    model_path = os.path.join(RealMapConfig.SAVE_DIR, "real_map_best_iou.pth")
    if not os.path.exists(model_path):
        print("Fine-tuned model not found, using pretrained synthetic model.")
        model_path = RealMapConfig.PRETRAINED_WEIGHTS
    model = load_model(model_path, device)

    # Берём первое изображение из валидационной папки
    val_dir = RealMapConfig.VAL_IMG_DIR
    if os.path.exists(val_dir) and os.listdir(val_dir):
        test_img = os.path.join(val_dir, os.listdir(val_dir)[0])
        prob = predict(test_img, model, device)
        visualize(test_img, prob, save_path="eval_result.png")
    else:
        print("No validation images found. Run generate_real_map_dataset.py first.")