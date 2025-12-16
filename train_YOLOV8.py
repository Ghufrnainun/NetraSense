from ultralytics import YOLO
import torch

# Cek GPU dulu biar yakin
print(f"Menggunakan Device: {torch.cuda.get_device_name(0)}")

def main():
    # 1. Load Model
    # Kita pake 'yolov8n.pt' (Nano) karena targetnya kecepatan real-time & HP kentang.
    # Kalau mau agak pinteran dikit pake 'yolov8s.pt' (Small).
    model = YOLO('yolov8n.pt')

    # 2. Training
    results = model.train(
        data='dataset/data.yaml',   # Pastikan path ini bener arah ke file yaml lu
        epochs=100,                 # 100 ronde cukup buat awal
        imgsz=640,                  # Ukuran gambar
        batch=16,                   # Batch size aman buat RTX 4050 (6GB VRAM)
        device=0,                   # GPU 0 (NVIDIA)
        name='yolov8_rupiah_2022',  # Nama folder hasil
        patience=20,                # Kalau 20 epoch gak ada kemajuan, stop otomatis (Early Stopping)
        workers=4                   # Biar CPU bantu nyiapin data
    )
    
    # 3. Validasi
    metrics = model.val()
    print("Training YOLOv8 Selesai! 🥳")

if __name__ == '__main__':
    # Wajib pake if __name__ == '__main__' di Windows biar multiprocessing gak error
    main()