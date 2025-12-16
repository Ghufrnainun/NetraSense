import cv2
import os
import glob

# === CONFIG ===
source_folder = "dataset_video"
output_folder = "dataset_mentah"
frame_interval = 10
# ==============

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Cari nomor terakhir ---
existing_files = sorted(glob.glob(os.path.join(output_folder, "*.jpg")))
if len(existing_files) > 0:
    last_file = os.path.basename(existing_files[-1]).split('.')[0]
    global_id = int(last_file)
else:
    global_id = 0

print(f"Start dari ID: {global_id}")
# ----------------------------

video_files = glob.glob(os.path.join(source_folder, "*.mp4"))

print(f"Ditemukan {len(video_files)} video. Gas proses! 🚀")

for video_path in video_files:
    filename = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    
    frame_id = 0
    print(f"--> Processing: {filename}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            save_name = f"{output_folder}/{global_id:05}.jpg"
            cv2.imwrite(save_name, frame)
            global_id += 1

        frame_id += 1

    cap.release()

print("\n✨ DONE My Lord ✨")
print(f"Total output sampe: {global_id} gambar 💖")
