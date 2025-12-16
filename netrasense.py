import cv2
import torch
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
import time
import threading
import pygame
from collections import deque

# ==========================================
# ⚙️ KONFIGURASI PROJEK
# ==========================================
MODEL_PATH = 'final_models/YoloV8.pt' # Pastikan path model bener
SOUND_FOLDER = "suara_cache"
COOLDOWN_TIME = 3.5

# -- SETTINGAN HUD (TAMPILAN) --
SIDEBAR_WIDTH = 300
COLOR_BG = (30, 30, 30)      # Dark Grey
COLOR_ACCENT = (0, 255, 217) # Cyan Neon
COLOR_TEXT = (255, 255, 255) # White

# -- SETTINGAN AKURASI --
CONFIDENCE_MAP = {
    0: 0.80, # 100k 
    1: 0.70, # 10k 
    2: 0.70, # 20k 
    3: 0.80, # 50k 
    4: 0.65, # 5k 
}
DEFAULT_CONF = 0.65

KAMUS_KELAS = {
    0: {"nama": "100 Ribu", "full": "Seratus Ribu Rupiah", "nominal": 100000, "warna": (0, 0, 200)},    
    1: {"nama": "10 Ribu", "full": "Sepuluh Ribu Rupiah", "nominal": 10000, "warna": (200, 0, 200)},   
    2: {"nama": "20 Ribu", "full": "Dua Puluh Ribu Rupiah", "nominal": 20000, "warna": (0, 200, 0)},   
    3: {"nama": "50 Ribu", "full": "Lima Puluh Ribu Rupiah", "nominal": 50000, "warna": (200, 0, 0)},   
    4: {"nama": "5 Ribu", "full": "Lima Ribu Rupiah", "nominal": 5000, "warna": (100, 150, 200)},     
}

# ==========================================
# 🛠️ INIT SYSTEM
# ==========================================
if not os.path.exists(SOUND_FOLDER): os.makedirs(SOUND_FOLDER)
pygame.mixer.init()
is_speaking = False
last_speak_time = 0
detection_history = deque(maxlen=8)

# ==========================================
# 🎨 FUNGSI UI (HUD)
# ==========================================
def draw_ui(frame, status_text, stable_data):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (SIDEBAR_WIDTH, h), COLOR_BG, -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    cv2.putText(frame, "NETRASENSE", (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, COLOR_ACCENT, 2)
    cv2.line(frame, (20, 60), (SIDEBAR_WIDTH - 20, 60), COLOR_TEXT, 1)

    color_status = (0, 255, 0) if "SIAP" in status_text else (0, 255, 255)
    if "GELAP" in status_text: color_status = (0, 165, 255) # Orange kalo gelap

    cv2.rectangle(frame, (20, 80), (SIDEBAR_WIDTH - 20, 130), color_status, -1)
    cv2.putText(frame, status_text, (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.putText(frame, "TOTAL DETEKSI:", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    nominal_display = "Rp 0"
    if stable_data:
        nominal_display = f"Rp {stable_data['total']:,}".replace(",", ".")
    cv2.putText(frame, nominal_display, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_TEXT, 3)

    cv2.putText(frame, "RINCIAN:", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    y_offset = 340
    if stable_data:
        for name in stable_data['names']:
            cv2.putText(frame, f"• {name}", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
            y_offset += 40
    else:
        cv2.putText(frame, "-", (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)
    
    return frame

# ==========================================
# 🧠 CORE LOGIC (YANG SUDAH DIGABUNG)
# ==========================================

# Kita gabung logic brightness check + CLAHE di sini biar rapi
def smart_preprocessing(frame):
    # 1. Cek dulu, gelap gak?
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    is_dark = False

    # Kalau brightness di bawah 90 (agak gelap), aktifkan CLAHE
    if brightness < 90:
        is_dark = True
        
        # --- LOGIC CLAHE MULAI ---
        # Convert ke LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE ke L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Gabung lagi
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # --- LOGIC CLAHE SELESAI ---

    return frame, is_dark

def play_audio_thread(text):
    global is_speaking
    is_speaking = True
    try:
        safe_name = text.replace(" ", "_").lower() + ".mp3"
        file_path = os.path.join(SOUND_FOLDER, safe_name)
        if not os.path.exists(file_path):
            tts = gTTS(text=text, lang='id')
            tts.save(file_path)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): time.sleep(0.1)
    except Exception as e:
        print(f"Error audio: {e}")
    finally: is_speaking = False

def get_position_text(x1, x2, width):
    center_obj = (x1 + x2) / 2
    if center_obj < width/3: return "di Kiri"
    elif center_obj > (width/3)*2: return "di Kanan"
    return ""

def analyze_stability():
    if len(detection_history) < 6: return None
    recent = list(detection_history)[-6:]
    first = recent[0]
    if first is None: return None
    for d in recent:
        if d is None or d['total'] != first['total']: return None
    return first

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def main():
    global last_speak_time
    print("🚀 NetraSense Pro UI: Loading...")
    
    # Load Model
    try:
        model = YOLO(MODEL_PATH)
    except:
        print(f"❌ Error: Model tidak ditemukan di {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(0) # Ganti 0 kalo pake webcam laptop
    cap.set(3, 1280); cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Preprocessing Cerdas (Auto Night Mode)
        # Ini otomatis nyalain CLAHE kalo gelap
        frame, is_dark_mode = smart_preprocessing(frame)

        # 2. Inference
        results = model(frame, stream=True, agnostic_nms=True, verbose=False)

        current_det = None
        frame_total = 0
        frame_names_short = []
        frame_names_full = []
        largest_area = 0
        pos_hint = ""
        detected = False
        boxes_to_draw = [] 

        for r in results:
            for box in r.boxes:
                # Filter Geometri
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                # Filter rasio (uang itu persegi panjang)
                # Kalo terlalu kotak/vertikal (kayak muka), skip
                if (w / h) < 1.1: continue 
                
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in KAMUS_KELAS:
                    # Ambil threshold khusus per kelas
                    threshold_required = CONFIDENCE_MAP.get(cls_id, DEFAULT_CONF)
                    
                    if conf >= threshold_required:
                        detected = True
                        data = KAMUS_KELAS[cls_id]
                        
                        area = w * h
                        if area > largest_area:
                            largest_area = area
                            pos_hint = get_position_text(x1, x2, 1280)

                        frame_total += data['nominal']
                        frame_names_short.append(data['nama'])
                        frame_names_full.append(data['full'])
                        
                        boxes_to_draw.append({
                            'coords': (x1, y1, x2, y2),
                            'color': data['warna'],
                            'label': f"{data['nama']} {int(conf*100)}%"
                        })

        # 3. Gambar Box
        for b in boxes_to_draw:
            cv2.rectangle(frame, (b['coords'][0], b['coords'][1]), (b['coords'][2], b['coords'][3]), b['color'], 2)
            cv2.putText(frame, b['label'], (b['coords'][0], b['coords'][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, b['color'], 2)

        # 4. Logic Validasi
        warning_msg = ""
        # Kalo gelap, kita kasih info di UI tapi gak usah ngomong terus2an
        ui_warning = "Mode Malam Aktif" if is_dark_mode else ""

        if detected:
            ratio = largest_area / (1280 * 720)
            if ratio < 0.03: 
                warning_msg = "Terlalu Jauh"
                detection_history.append(None)
            elif ratio > 0.80: # Sedikit longgarin biar ga bawel
                warning_msg = "Terlalu Dekat"
                detection_history.append(None)
            else:
                detection_history.append({'total': frame_total, 'names': frame_names_short, 'full_names': frame_names_full, 'pos': pos_hint})
        else:
            detection_history.append(None)

        # 5. UI Logic
        stable_data = analyze_stability()
        
        status_text = "SCANNING..."
        if warning_msg: status_text = warning_msg.upper()
        elif ui_warning: status_text = ui_warning.upper() # Tampilkan "MODE MALAM" kalo gelap tapi ga ada warning jarak
        elif stable_data: status_text = "SIAP BICARA"
        elif not detected: status_text = "MENCARI..."

        frame = draw_ui(frame, status_text, stable_data)

        # 6. Audio Logic
        curr_time = time.time()
        
        # Audio Warning (Jauh/Dekat)
        if (warning_msg == "Terlalu Jauh" or warning_msg == "Terlalu Dekat") and not is_speaking and (curr_time - last_speak_time > 3.0):
            print(f"⚠️ {warning_msg}")
            threading.Thread(target=play_audio_thread, args=(warning_msg,)).start()
            last_speak_time = curr_time
            
        # Audio Nominal
        elif stable_data and not is_speaking and (curr_time - last_speak_time > COOLDOWN_TIME):
            total = stable_data['total']
            names_full = stable_data['full_names']
            pos = stable_data['pos']
            
            if len(names_full) == 1:
                kalimat = f"{names_full[0]} {pos}"
            else:
                kalimat = f"Total {total} Rupiah"
            
            print(f"🔊 {kalimat}")
            threading.Thread(target=play_audio_thread, args=(kalimat,)).start()
            last_speak_time = curr_time
            detection_history.clear()

        cv2.imshow("NetraSense Pro HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()