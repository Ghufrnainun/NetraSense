import torch
import cv2
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# === KONFIGURASI ===
MODEL_PATH = "final_models/faster_rcnn_best.pth"  # Pastikan path ini bener
TEST_DIR = "dataset/test"       # Kita uji pake data TEST
NUM_CLASSES = 6                 # 4 Uang + 1 Background
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# === 1. DATASET LOADER (Copas dikit dari training biar jalan) ===
class SimpleTestDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.imgs[idx].replace(".jpg", ".txt"))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img_tensor = F.to_tensor(img)

        boxes = []
        labels = []
        
        # Baca Label buat hitung akurasi (Ground Truth)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0]) + 1
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    x_min = (cx - bw/2) * w
                    y_min = (cy - bh/2) * h
                    x_max = (cx + bw/2) * w
                    y_max = (cy + bh/2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(cls_id)

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        target["labels"] = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0), dtype=torch.int64)
        
        return img_tensor, target, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

# === 2. FUNGSI UTAMA ===
def main():
    print(f"🧐 Memulai Evaluasi Faster R-CNN di {DEVICE}...")
    
    # Load Data
    dataset_test = SimpleTestDataset(TEST_DIR)
    data_loader = DataLoader(dataset_test, batch_size=4, collate_fn=collate_fn)
    
    # Load Model
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    # Load Bobot yang udah lu training
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Model berhasil dimuat!")
    else:
        print("❌ Model gak ketemu! Cek path-nya.")
        return

    model.to(DEVICE)
    model.eval()
    
    # Siapkan Kalkulator mAP
    metric = MeanAveragePrecision(iou_type="bbox")
    
    # Siapkan Folder Output Gambar
    if not os.path.exists("hasil_visual_rcnn"):
        os.makedirs("hasil_visual_rcnn")

    print("🚀 Sedang menghitung mAP & FPS... (Agak lama, tungguin)")
    
    start_time = time.time()
    total_frames = 0
    
    with torch.no_grad():
        for i, (images, targets, filenames) in enumerate(data_loader):
            images = list(img.to(DEVICE) for img in images)
            
            # INFERENCE (Tebak Gambar)
            preds = model(images)
            
            # Masukin ke kalkulator mAP
            # Kita harus format ulang dikit biar torchmetrics ngerti
            metric_preds = []
            metric_targets = []
            
            for p, t in zip(preds, targets):
                metric_preds.append(dict(boxes=p['boxes'].cpu(), scores=p['scores'].cpu(), labels=p['labels'].cpu()))
                metric_targets.append(dict(boxes=t['boxes'], labels=t['labels']))
            
            metric.update(metric_preds, metric_targets)
            
            # Hitung FPS
            total_frames += len(images)
            
            # --- SIMPAN CONTOH GAMBAR (Buat Laporan) ---
            # Kita simpan batch pertama aja biar laptop gak penuh
            if i == 0: 
                for j, img_tensor in enumerate(images):
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy().copy()
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Gambar Kotak Prediksi
                    boxes = preds[j]['boxes'].cpu().numpy()
                    scores = preds[j]['scores'].cpu().numpy()
                    
                    for k, box in enumerate(boxes):
                        if scores[k] > 0.5: # Cuma gambar yang yakin > 50%
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_np, f"{scores[k]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.imwrite(f"hasil_visual_rcnn/test_{j}.jpg", img_np)

    end_time = time.time()
    fps = total_frames / (end_time - start_time)
    
    # Hitung Final mAP
    result = metric.compute()
    
    print("\n" + "="*30)
    print("📊 HASIL EVALUASI FASTER R-CNN")
    print("="*30)
    print(f"⚡ Kecepatan (FPS) : {fps:.2f} frames/detik")
    print(f"🎯 mAP@50          : {result['map_50'].item():.4f}")
    print(f"🎯 mAP@50-95       : {result['map'].item():.4f}")
    print("="*30)
    print("✅ Cek folder 'hasil_visual_rcnn' buat liat contoh gambarnya!")

if __name__ == '__main__':
    main()