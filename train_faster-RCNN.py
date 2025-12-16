# ============================================
# Faster R-CNN Turbo + Auto-Resume (FIXED)
# ============================================

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import traceback
import time

TRIALS = [
    {"TARGET_SHORT_SIDE": 800, "BATCH_SIZE": 8, "AMP": True},
    {"TARGET_SHORT_SIDE": 800, "BATCH_SIZE": 4, "AMP": True},
    {"TARGET_SHORT_SIDE": 640, "BATCH_SIZE": 4, "AMP": True},
]

NUM_EPOCHS = 10
NUM_CLASSES = 6
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SIZE = 1333

CHECKPOINT_DIR = "checkpoints"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True


# ---------------------- DATASET ----------------------
class RupiahYOLODataset(Dataset):
    def __init__(self, root, target_short_side):
        self.root = root
        self.target_short_side = target_short_side

        img_dir = os.path.join(root, "images")
        self.imgs = [f for f in sorted(os.listdir(img_dir))
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "images", img_name)
        label_path = os.path.join(self.root, "labels", img_name.rsplit(".", 1)[0] + ".txt")

        img_arr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = self.target_short_side / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        if max(new_h, new_w) > MAX_SIZE:
            s = MAX_SIZE / max(new_h, new_w)
            new_w, new_h = int(new_w * s), int(new_h * s)

        img = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w

        img_tensor = F.to_tensor(img)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            for line in open(label_path, "r"):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cid = int(parts[0]) + 1
                xc, yc, bw, bh = map(float, parts[1:5])

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cid)

        # ----------- FIX: jika kosong, kasih tensor kosong shape Nx4 ----------
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------- CHECKPOINT ----------------------
def save_checkpoint(model, optimizer, scaler, epoch):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, LATEST_CKPT)
    print(f"[CKPT] Saved at epoch {epoch}")


def load_checkpoint(model, optimizer, scaler):
    if not os.path.exists(LATEST_CKPT):
        return 0

    print("[CKPT] Resume enabled. Loading checkpoint...")
    ckpt = torch.load(LATEST_CKPT, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"]


# ---------------------- TRAINING ----------------------
def run_trial(cfg):
    try:
        print(f"\n🔥 Trial: {cfg}")

        ds = RupiahYOLODataset("dataset/train", cfg["TARGET_SHORT_SIDE"])
        loader = DataLoader(ds,
                            batch_size=cfg["BATCH_SIZE"],
                            shuffle=True,
                            num_workers=0,
                            collate_fn=collate_fn)

        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        model.to(DEVICE)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LEARNING_RATE,
                                    momentum=0.9,
                                    weight_decay=0.0005)

        scaler = torch.amp.GradScaler("cuda", enabled=cfg["AMP"])
        start_epoch = load_checkpoint(model, optimizer, scaler)

        print(f"✨ Starting at epoch {start_epoch}")

        # -------- TRAIN LOOP --------
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()

            for i, (images, targets) in enumerate(loader):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                with torch.amp.autocast("cuda", enabled=cfg["AMP"]):
                    loss_dict = model(images, targets)
                    loss = sum(v for v in loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if i % 30 == 0:
                    print(f"[Ep {epoch+1}] Step {i} Loss {loss.item():.4f}")

            save_checkpoint(model, optimizer, scaler, epoch + 1)

        # .... setelah loop epoch selesai ....
        
        # Save Final Model terpisah
        torch.save(model.state_dict(), "final_models/faster_rcnn_best.pth")
        print("🎉 Final Model Saved to final_models/faster_rcnn_best.pth")
        
        print("✔ Training Done!")
        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠ OOM detected, switching config…")
            torch.cuda.empty_cache()
            return False
        raise e


# ---------------------- MAIN ----------------------
def main():
    for cfg in TRIALS:
        if run_trial(cfg):
            return
    print("❌ Semua config gagal.")


if __name__ == "__main__":
    main()
