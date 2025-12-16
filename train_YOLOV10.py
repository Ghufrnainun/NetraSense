from ultralytics import YOLO
import torch

def main():
    # --- BAGIAN CEK GPU (Ditaruh di dalam fungsi biar aman) ---
    if torch.cuda.is_available():
        print(f"🔥 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print("Siap menyiksa RTX 4050!")
    else:
        print("⚠️ WARNING: Masih pake CPU! Cek installan PyTorch lu.")
    # -----------------------------------------------------------

    # 1. Load Model YOLOv10 Nano
    # Script ini bakal otomatis download 'yolov10n.pt' kalau belum ada.
    # YOLOv10n (Nano) dipilih biar adil perbandingannya sama YOLOv8n.
    print("Sedang memuat model YOLOv10...")
    model = YOLO('yolov10n.pt')

    # 2. Mulai Training
    print("🚀 Mulai Training YOLOv10...")
    results = model.train(
        data='dataset/data.yaml',   # Pastikan path ini bener
        epochs=100,                 # Samain kayak YOLOv8 kemarin
        imgsz=640,
        batch=16,                   # RTX 4050 kuat nampung ini
        device=0,                   # Paksa pake GPU NVIDIA
        name='yolov10_rupiah_2022', # Nama folder output beda biar gak ketimpa
        patience=20,                # Early stopping
        workers=2,                  # PENTING: Set 2 atau 1 aja di Windows biar gak crash memori
        exist_ok=True               # Kalau folder udah ada, timpa aja (biar gak error mkdir)
    )
    
    print("✅ Training YOLOv10 Selesai! Silakan cek folder runs/detect.")

if __name__ == '__main__':
    # INI JIMAT ANTI-ERROR DI WINDOWS
    # Semua perintah eksekusi HARUS ada di bawah blok ini.
    import multiprocessing
    multiprocessing.freeze_support() # Tambahan pengaman
    main()