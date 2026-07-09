import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Ограничиваем потоки CPU для OpenBLAS/OpenCV
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

# ============================== КОНФИГУРАЦИЯ ==============================
CFG = {
    "train_csv": "stage1/train.csv",
    "output_dir": "output",
    "image_size": 224,
    "batch_size": 4,
    "num_workers": 0,
    "accumulation_steps": 4,
    "epochs": 10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "encoder": "efficientnet-b0",
    "encoder_weights": "imagenet",
    "use_amp": True,
    "bce_weight": 1.0,
    "dice_weight": 1.0,
    "val_split": 0.2,
    "val_subset_size": 3000,
    "min_contour_area": 10,
    "seed": 42,
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(CFG["output_dir"], exist_ok=True)

# ============================== DATASET ==============================
class ForgeryDataset(Dataset):
    def __init__(self, df, image_size=320, augment=True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["chng_img_path"]
        mask_path = row["gt_path"]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            mask = (mask > 127).astype(np.uint8)
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.1:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            k = random.choice([0, 1, 2, 3])
            if k != 0:
                img = TF.rotate(img, 90 * k, expand=False)
                mask = TF.rotate(mask, 90 * k, expand=False)
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
                img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))

        return img, mask

# ============================== МОДЕЛЬ ==============================
def build_model():
    model = smp.Unet(
        encoder_name=CFG["encoder"],
        encoder_weights=CFG["encoder_weights"],
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model

# ============================== ФУНКЦИИ ПОТЕРЬ ==============================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()

def criterion(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d = dice_loss(pred, target)
    return CFG["bce_weight"] * bce + CFG["dice_weight"] * d

# ============================== МЕТРИКИ ==============================
def compute_dice(pred_mask, gt_mask):
    smooth = 1.0
    intersection = (pred_mask * gt_mask).sum()
    return (2.0 * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)

def remove_small_components(mask_bin, min_area):
    if min_area <= 0:
        return mask_bin
    mask_uint = (mask_bin * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            labels[labels == i] = 0
    return (labels > 0).astype(np.uint8)

def evaluate_model_and_find_threshold(model, val_loader, device, min_area, subset_for_thresh):
    model.eval()
    pred_probs_sub = []
    gt_masks_sub = []
    count = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            out = model(images)
            probs = torch.sigmoid(out).cpu().numpy()
            masks_np = masks.cpu().numpy().squeeze(1)
            for prob, mask in zip(probs, masks_np):
                if count >= subset_for_thresh:
                    break
                pred_probs_sub.append(prob.squeeze(0))
                gt_masks_sub.append(mask)
                count += 1
            if count >= subset_for_thresh:
                break

    best_thresh, _ = find_best_threshold(pred_probs_sub, gt_masks_sub, min_area)
    del pred_probs_sub, gt_masks_sub
    torch.cuda.empty_cache()

    total_dice_pos = 0.0
    count_pos = 0
    fp_count = 0
    count_neg = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            out = model(images)
            probs = torch.sigmoid(out).cpu().numpy()
            masks_np = masks.cpu().numpy().squeeze(1)

            for prob, gt_mask in zip(probs, masks_np):
                prob = prob.squeeze(0)
                bin_mask = (prob >= best_thresh).astype(np.uint8)
                if min_area > 0:
                    bin_mask = remove_small_components(bin_mask, min_area)

                if gt_mask.max() > 0:
                    d = compute_dice(torch.tensor(bin_mask), torch.tensor(gt_mask)).item()
                    total_dice_pos += d
                    count_pos += 1
                else:
                    if bin_mask.max() > 0:
                        fp_count += 1
                    count_neg += 1

    dice_pos = total_dice_pos / count_pos if count_pos > 0 else 0.0
    fpr = fp_count / count_neg if count_neg > 0 else 0.0
    if dice_pos + (1 - fpr) == 0:
        aic = 0.0
    else:
        aic = 2 * dice_pos * (1 - fpr) / (dice_pos + (1 - fpr))
    print(f"Порог: {best_thresh:.2f}, Val Dice (pos): {dice_pos:.4f}, FPR (neg): {fpr:.4f}, AIC Score: {aic:.4f}")
    return best_thresh, aic, dice_pos, fpr

def find_best_threshold(pred_probs, gt_masks, min_area):
    best_score = -1
    best_thresh = 0.5
    pos_indices = [i for i, m in enumerate(gt_masks) if m.max() > 0]
    neg_indices = [i for i, m in enumerate(gt_masks) if m.max() == 0]
    if not pos_indices or not neg_indices:
        return 0.5, 0.0
    thresholds = np.linspace(0, 1, 101)
    for th in thresholds:
        bin_masks = [(prob >= th).astype(np.uint8) for prob in pred_probs]
        if min_area > 0:
            bin_masks = [remove_small_components(bm, min_area) for bm in bin_masks]

        dice_pos = [compute_dice(torch.tensor(bin_masks[i]), torch.tensor(gt_masks[i])).item() for i in pos_indices]
        mean_dice = np.mean(dice_pos)

        fp_count = sum(1 for i in neg_indices if bin_masks[i].max() > 0)
        fpr = fp_count / len(neg_indices)

        if mean_dice + (1 - fpr) == 0:
            score = 0.0
        else:
            score = 2 * mean_dice * (1 - fpr) / (mean_dice + (1 - fpr))
        if score > best_score:
            best_score = score
            best_thresh = th
    return best_thresh, best_score

# ============================== TRAIN LOOP ==============================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Training")
    for i, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)

        with torch.cuda.amp.autocast(enabled=CFG["use_amp"]):
            preds = model(images)
            loss = criterion(preds, masks) / CFG["accumulation_steps"]

        scaler.scale(loss).backward()

        if (i + 1) % CFG["accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * CFG["accumulation_steps"]
        loop.set_postfix(loss=running_loss/(i+1))
    return running_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(CFG["train_csv"])
    df = df.dropna(subset=["chng_img_path", "gt_path"])

    train_df, val_df = train_test_split(df, test_size=CFG["val_split"], random_state=CFG["seed"])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = ForgeryDataset(train_df, image_size=CFG["image_size"], augment=True)
    val_dataset = ForgeryDataset(val_df, image_size=CFG["image_size"], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False,
                            num_workers=CFG["num_workers"], pin_memory=False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["use_amp"])

    best_score = -1.0
    best_thresh = 0.5

    for epoch in range(1, CFG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CFG['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}")

        best_thresh, score, dice_pos, fpr = evaluate_model_and_find_threshold(
            model, val_loader, device, CFG["min_contour_area"],
            subset_for_thresh=CFG["val_subset_size"]
        )

        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "best_threshold": best_thresh,
                "config": CFG,
            }, os.path.join(CFG["output_dir"], "best_model.pth"))
            print("Сохранили лучшую модель.")

        torch.cuda.empty_cache()

    print(f"\nОбучение завершено. Лучший AIC Score на валидации: {best_score:.4f}")

if __name__ == "__main__":
    main()