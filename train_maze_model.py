import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

scaler = GradScaler(device="cuda")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# --- Датасет ---
class MazeDataset(Dataset):
    def __init__(self, mazes_folder, solutions_folder, filelist_cache="filelist_77x77.txt"):
        self.mazes_folder = mazes_folder
        self.solutions_folder = solutions_folder
        self.files = []
        t0 = time.time()
        if os.path.exists(filelist_cache):
            with open(filelist_cache, "r") as f:
                self.files = [line.strip() for line in f]
            print(f"Загружено {len(self.files)} файлов из кэша за {time.time()-t0:.1f} сек.")
        else:
            for f in os.listdir(mazes_folder):
                if not f.endswith('.npy'):
                    continue
                sol_path = os.path.join(solutions_folder, f)
                if os.path.exists(sol_path):
                    self.files.append(f)
            with open(filelist_cache, "w") as f:
                for fname in self.files:
                    f.write(fname + "\n")
            print(f"Список файлов ({len(self.files)}) сохранён в кэш за {time.time()-t0:.1f} сек.")
        if not self.files:
            raise RuntimeError(f"Нет файлов в {mazes_folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        maze = np.load(os.path.join(self.mazes_folder, fname))
        solution = np.load(os.path.join(self.solutions_folder, fname))
        maze = 1 - maze  # инверсия: проходы=1, стены=0
        maze = maze.astype(np.uint8)  # гарантируем тип
        solution = solution.astype(np.uint8)
        # Для отладки: выводим уникальные значения
        # print("maze_inv unique values:", np.unique(maze))
        # print("solution unique values:", np.unique(solution))
        maze = torch.tensor(maze, dtype=torch.float32).unsqueeze(0)
        solution = torch.tensor(solution, dtype=torch.float32).unsqueeze(0)
        return maze, solution

# --- U-Net модель ---
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
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
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
        )
        # Up part (fix: track in_channels as you go up)
        up_in_channels = features[-1]*2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(up_in_channels, feature, kernel_size=2, stride=2)
            )
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(feature*2, feature, 3, padding=1),
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

    def forward(self, x):
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
            x = self.ups[idx+1](x)
        return self.final_conv(x)

# --- Обучение ---
def train(continue_from_checkpoint=False, checkpoint_path="maze_model_best_f1.pth"):
    mazes_folder = "mazes"
    solutions_folder = "solutions"
    dataset = MazeDataset(mazes_folder, solutions_folder)
    print("Файлов для обучения:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=64, 
        shuffle=True,
        num_workers=4
    )
    if torch.cuda.is_available():
        print("CUDA доступен. Используется GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA не доступен. Используется CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Используемое устройство:", device)
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # --- ДОБАВЛЕНО: загрузка чекпоинта для продолжения обучения ---
    best_f1 = 0.0
    if continue_from_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Загружена модель из {checkpoint_path} для продолжения обучения.")
    else:
        torch.save(model.state_dict(), "maze_model.pth")

    losses = []
    accs = []
    precs = []
    recs = []
    f1s = []

    # best_f1 = 0.0

    # --- Подсчёт pos_weight для баланса классов ---
    sample_solution = np.load(os.path.join(solutions_folder, dataset.files[0]))
    pos = np.sum(sample_solution)
    neg = sample_solution.size - pos
    pos_weight = torch.tensor([400.0], dtype=torch.float32, device=device)
    print(f"pos_weight для BCEWithLogitsLoss: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Начинаем обучение...")
    for epoch in range(50):  # только 20 эпох
        print(f"\n========== Эпоха {epoch+1} ==========")
        model.train()
        total_loss = 0
        acc_sum = 0
        prec_sum = 0
        rec_sum = 0
        f1_sum = 0
        n_batches = 0
        for batch_idx, batch in enumerate(loader):
            maze, solution = batch
            maze = maze.to(device)
            solution = solution.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(maze)
                loss = criterion(output, solution)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).detach().cpu().numpy().astype(int).flatten()
            trues = solution.detach().cpu().numpy().astype(int).flatten()
            acc_sum += accuracy_score(trues, preds)
            prec_sum += precision_score(trues, preds, zero_division=0)
            rec_sum += recall_score(trues, preds, zero_division=0)
            f1_sum += f1_score(trues, preds, zero_division=0)
            n_batches += 1
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                print(f"  Batch {batch_idx+1}/{len(loader)} | loss: {loss.item():.4f}")

        acc = acc_sum / n_batches
        prec = prec_sum / n_batches
        rec = rec_sum / n_batches
        f1 = f1_sum / n_batches
        losses.append(total_loss / n_batches)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        print(f"Epoch {epoch+1}, loss: {total_loss/n_batches:.4f} | acc: {acc:.3f} | prec: {prec:.3f} | rec: {rec:.3f} | f1: {f1:.3f}")

        # Early model checkpointing по лучшему f1
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "maze_model_best_f1.pth")
            print(f"Сохранена новая лучшая модель (f1={f1:.4f})")

        # --- отладка: выводим min/max/mean по выходу модели для одного батча ---
        if epoch == 0 and batch_idx == 0:
            out_np = torch.sigmoid(output).detach().cpu().numpy()
            print("output sigmoid min:", out_np.min(), "max:", out_np.max(), "mean:", out_np.mean())
            print("solution min:", solution.min().item(), "max:", solution.max().item(), "mean:", solution.float().mean().item())
            print("maze min:", maze.min().item(), "max:", maze.max().item(), "mean:", maze.float().mean().item())
    print("Обучение завершено.")

    # --- Визуализация ---
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(losses, label='Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(accs, label='Accuracy')
    plt.plot(precs, label='Precision')
    plt.plot(recs, label='Recall')
    plt.plot(f1s, label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Для продолжения обучения с чекпоинта:
    train(continue_from_checkpoint=True, checkpoint_path="maze_model_best_f1.pth")
    # Для обучения с нуля:
    # train()

# Рекомендации по дообучению:
# 1. Если f1-score на обучении всё ещё растёт — дообучайте ещё 10–20 эпох.
# 2. Если f1-score "застыл", попробуйте:
#    - уменьшить learning rate (например, до 1e-6)
#    - увеличить pos_weight (например, до 500)
#    - добавить Dropout(0.2) после ReLU в UNet
#    - обучать с early stopping: если f1 не растёт 5 эпох — остановить
# 3. Если модель переобучается (f1 падает) — уменьшите количество эпох или используйте Dropout.

# Практически:
# - Обычно до 40–60 эпох для такой задачи достаточно.
# - Если метрики растут — продолжайте, если нет — меняйте параметры.