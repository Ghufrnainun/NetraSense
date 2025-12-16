from ultralytics import YOLO
import torch
import multiprocessing

def main():
    # 1. Cek GPU (Opsional, buat gaya aja)
    if torch.cuda.is_available():
        print(f"🔥 Lanjutin Training pake: {torch.cuda.get_device_name(0)}")

    # 2. LOAD CHECKPOINT TERAKHIR (last.pt)
    # Ini kuncinya. Kita panggil 'last.pt' biar dia inget dia mati di epoch 90.
    # Pastikan path-nya bener sesuai struktur folder lu di screenshot tadi.
    model = YOLO('runs/detect/yolov8_rupiah_2022/weights/last.pt') 

    # 3. RESUME TRAINING
    print("🚀 Melanjutkan sisa epoch yang tertunda...")
    results = model.train(
        resume=True,    # PERINTAH AJAIB: Lanjutin, jangan ulang!
        workers=2       # PENTING: Gw set 2 biar RAM lu gak meledak lagi kayak tadi
    )
    
    print("✅ SELESAI FINISH 100%! Cek folder runs sekarang.")

if __name__ == '__main__':
    # Proteksi wajib buat Windows
    multiprocessing.freeze_support()
    main()