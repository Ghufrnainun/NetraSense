import torch
print(f"Pytorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("MAMPUS, MASIH PAKE CPU BRO! SALAH INSTALL.")