# 👁️ NetraSense: Rupiah Audio-Identifier

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v10-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20RTX%20Enabled-blueviolet)

> **"Seeing Value through Sound."** > An assistive technology system designed to help visually impaired individuals identify Indonesian Rupiah banknotes using Deep Learning and Real-time Audio Feedback.

---

## 📋 Table of Contents
- [Background](#-background)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Dataset Strategy](#-dataset-strategy)
- [Model Comparison](#-model-performance--comparison)
- [Installation & Usage](#-installation--usage)
- [Hardware Used](#-hardware-used)
- [Author](#-author)

---

## 📖 Background

While Bank Indonesia includes "Blind Codes" (tactile lines) on banknotes, these features often degrade due to circulation wear (crumpled/old money), making them difficult for the visually impaired to identify by touch.
**NetraSense** solves this by using Computer Vision to "see" the visual patterns of the money—regardless of its physical texture—and converts the detection into **Audio Feedback** (Speech).
This project focuses on Indonesian Rupiah, specifically bridging the gap where traditional tactile methods fail.

---

## 🌟 Key Features

* **🗣️ Smart Audio Feedback:** Converts visual detections into natural speech (Indonesian Language) using Google Text-to-Speech (gTTS).
* **💰 Total Summation Logic:** Capable of detecting multiple banknotes at once and calculating the **Total Value** (e.g., "Total: Rp 65,000").
* **🧠 Multi-Model Architecture:** Implements and compares three different architectures:
    * **YOLOv8** (Balanced Performance)
    * **YOLOv10** (NMS-Free / Low Latency)
    * **Faster R-CNN** (High Precision Benchmark)
* **🛑 Anti-Spam Cooldown:** Intelligent logic to prevent the audio from repeating continuously, ensuring a pleasant user experience.
* **⚡ Threaded Processing:** Audio generation runs on a separate thread to ensure the video feed remains smooth (60 FPS).

---
## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.x | Core logic |
| **CV Engine** | OpenCV | Image processing & Webcam stream |
| **AI Models** | Ultralytics YOLO & PyTorch | Object Detection backbone |
| **Audio** | gTTS & Pygame | Text-to-Speech generation & playback |
| **Dataset** | Roboflow | Annotation, Split, and Augmentation |

---

## 📸 Dataset Strategy

The dataset was collected manually to simulate real-world conditions faced by visually impaired users.

* **Classes:** 5 Classes (`5k`, `10k`, `20k`, `50k`, `100k`).
* **Total Images:** 1,500+ (Raw) -> 4,500+ (Augmented).
* **Scenarios:**
    * ✅ **Flat lay:** Ideal condition.
    * ✅ **Hand-held:** Simulating transaction gestures.
    * ✅ **Occluded/Crumpled:** Simulating worn-out money.
    * ✅ **Low Light:** Indoor room environment.
* **Augmentation:** Shear, Rotation (±15°), Brightness, and Noise Injection.

---

## 📊 Model Performance & Comparison

This project analyzes the trade-off between Accuracy (mAP) and Speed (FPS) for assistive devices.

| Model Architecture | mAP@50 (Accuracy) | Inference Speed (FPS) | Model Size | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **YOLOv8 Nano** | **99.1%** | **45 FPS** | 6.2 MB | 🏆 **Best Overall** |
| **YOLOv10 Nano** | 98.5% | 50 FPS | 5.8 MB | ⚡ Fastest |
| **Faster R-CNN** | 99.4% | 8 FPS | 100+ MB | 🐢 Too Slow |

*> **Note:** Tested on NVIDIA RTX 4050 Laptop GPU.*

---

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/username/NetraSense.git](https://github.com/username/NetraSense.git)
    cd NetraSense
    ```

2.  **Create Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install ultralytics opencv-python pygame gTTS torch torchvision
    ```
    *(Ensure you have PyTorch with CUDA support if you have an NVIDIA GPU)*

4.  **Run the Application**
    ```bash
    python app_demo.py
    ```

---

## 💻 Hardware Used

This project was developed and trained on:
* **CPU:** AMD Ryzen 7 8845HS
* **GPU:** NVIDIA GeForce RTX 4050 (6GB VRAM)
* **RAM:** 16 GB DDR5

---

## 👨‍💻 Author

**Ghufron Ainun Najib** *Computer Engineering Student at Politeknik Negeri Semarang (Polines)* 
*"Building technology that matters."*

---
