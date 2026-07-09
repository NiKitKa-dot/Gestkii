import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import zipfile

CFG = {
    "test_csv": "test_stage1/test.csv",
    "model_path": "output/best_model.pth",
    "output_dir": "predictions",
    "image_size": 224,            # строго как в обучении
    "batch_size": 4,
    "num_workers": 0,
    "encoder": "efficientnet-b0",
    "submission_zip": "submission.zip",
    "base_img_dir": "test_stage1"   # где лежит test.csv и папка test_stage1_img
}

os.makedirs(CFG["output_dir"], exist_ok=True)

class TestDataset(Dataset):
    def __init__(self, df, image_size, base_img_dir):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.base_img_dir = base_img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        relative_path = row["img_path"]                          # напр. "test_stage1_img/img.jpg"
        full_img_path = os.path.join(self.base_img_dir, relative_path)  # test_stage1/test_stage1_img/img.jpg

        original_img = cv2.imread(full_img_path)
        if original_img is None:
            raise FileNotFoundError(f"Не найдено изображение: {full_img_path}")
        orig_h, orig_w = original_img.shape[:2]

        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Возвращаем относительный путь как в исходном test.csv
        return img, relative_path, orig_h, orig_w

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Загрузка модели
    model = smp.Unet(
        encoder_name=CFG["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    checkpoint = torch.load(CFG["model_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"]
    print(f"Модель загружена. Порог бинаризации: {best_threshold:.3f}")
    model.eval()

    # Чтение тестового CSV
    test_df = pd.read_csv(CFG["test_csv"])
    print(f"Найдено {len(test_df)} тестовых изображений")

    test_dataset = TestDataset(test_df, CFG["image_size"], CFG["base_img_dir"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=False,
    )

    submission_rows = []

    with torch.no_grad():
        for images, rel_paths, orig_h, orig_w in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for prob, rel_path, h, w in zip(probs, rel_paths, orig_h, orig_w):
                prob = prob.squeeze(0)
                bin_mask = (prob >= best_threshold).astype(np.uint8)

                # Ресайз маски до исходного размера
                bin_mask_resized = cv2.resize(bin_mask, (w.item(), h.item()), interpolation=cv2.INTER_NEAREST)

                # Имя файла маски
                fname = os.path.basename(rel_path)
                mask_filename = f"{os.path.splitext(fname)[0]}_mask.png"
                mask_path = os.path.join(CFG["output_dir"], mask_filename)
                cv2.imwrite(mask_path, bin_mask_resized * 255)

                # Путь в сабмите: исходный относительный путь (без base_img_dir)
                submission_rows.append({
                    "img_path": rel_path,                                # как в test.csv
                    "prediction_path": f"predictions/{mask_filename}"
                })

    # Сохранение submission.csv
    submission_df = pd.DataFrame(submission_rows)
    submission_csv_path = "submission.csv"
    submission_df.to_csv(submission_csv_path, index=False)
    print(f"Сохранён {submission_csv_path}")

    # Упаковка в zip
    with zipfile.ZipFile(CFG["submission_zip"], 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(submission_csv_path)
        for root, _, files in os.walk(CFG["output_dir"]):
            for file in files:
                full = os.path.join(root, file)
                arcname = os.path.relpath(full, start=".")
                zipf.write(full, arcname)
    print(f"Архив {CFG['submission_zip']} готов для отправки.")

if __name__ == "__main__":
    main()