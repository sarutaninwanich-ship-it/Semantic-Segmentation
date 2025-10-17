# Semantic-Segmentation

> **TH/EN README** — โปรเจกต์ Semantic Segmentation สำหรับโหลดข้อมูลจาก **Roboflow** และฝึกโมเดล พร้อมสคริปต์ **Gradio** สำหรับรันเว็บเดโม่แบบใช้งานได้ทันที

---

## 📦 Repository Overview

- `main.py` — ไฟล์หลัก: โหลด **dataset** จาก Roboflow ตามตัวแปรใน `.env` และ **train** โมเดล
- `Gradio/`
  - `_classes.csv` — รายชื่อคลาส (ลำดับต้องตรงกับโมเดล)
  - `app_semseg.py` — เว็บแอป Gradio สำหรับ inference/เดโม่
  - `best_lraspp_mbv3.pth` — ไฟล์ **weights** ของโมเดล (LRASPP + MobileNetV3 Large)
- `.env` — กำหนดค่า Roboflow และพารามิเตอร์เทรนต่าง ๆ (ตัวอย่างด้านล่าง)

โครงสร้างโดยรวม:
```
Semantic-Segmentation/
├─ main.py
├─ .env
└─ Gradio/
   ├─ _classes.csv
   ├─ app_semseg.py
   └─ best_lraspp_mbv3.pth
```

---

## ✅ Requirements / การติดตั้งไลบรารี

แนะนำติดตั้งด้วยคำสั่งด้านล่าง (รองรับทั้งโลคัลและ Google Colab):
```bash
pip install -q torch torchvision roboflow python-dotenv tqdm gradio pillow opencv-python
```

> หากใช้ CUDA ให้ติดตั้ง PyTorch ตามคู่มือเวอร์ชันและไดรเวอร์ของเครื่อง: https://pytorch.org/get-started/locally/

---

## 🔧 Environment (.env)

สร้างไฟล์ `.env` ที่ root และตั้งค่าตัวแปรดังนี้ (ตัวอย่าง):
```env
# --- Roboflow ---
RF_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RF_WORKSPACE=hoi-rmqtp
RF_PROJECT=human-object-interaction-pcpk1
RF_VERSION=2

# --- Training config (ปรับได้) ---
IMGSZ=640
BATCH=8
EPOCHS=20
LR=0.0003
ACCUM=4
AMP=1
PRELOAD=1
FREEZE_BACKBONE_EPOCHS=1
NUM_WORKERS=2

# โมเดล backbone/สถาปัตยกรรม (ตัวอย่าง)
MODEL=deeplabv3_resnet50
COMPILE=0
USE_TQDM=1

# Logging
LOG_EVERY=10
LOG_FILE=runs/semseg/train_log.csv
```

**หมายเหตุ**
- ถ้า dataset เป็น **Semantic Segmentation** ให้มั่นใจว่า pipeline/สคริปต์ใน `main.py` map กับรูปแบบ mask และจำนวนคลาสถูกต้อง
- ค่า `RF_*` จะใช้สำหรับดาวน์โหลด dataset อัตโนมัติจาก Roboflow

---

## 🚀 Quick Start

### 1) Train
รันที่โฟลเดอร์โปรเจกต์เดียวกันกับ `main.py`
```bash
python main.py
```
สคริปต์จะ:
1. โหลดค่า config จาก `.env`
2. ดาวน์โหลด dataset จาก Roboflow
3. เริ่มเทรนโมเดล และบันทึก log + checkpoints

> ผลลัพธ์ (weights/logs) จะเก็บใน `runs/` (หรือที่กำหนดใน `LOG_FILE`/สคริปต์)

### 2) Inference Demo (Gradio)
เข้าโฟลเดอร์ `Gradio/` แล้วรัน:
```bash
cd Gradio
python app_semseg.py
```
จากนั้นเปิดลิงก์ที่แสดงในคอนโซล (เช่น `http://127.0.0.1:7860`) เพื่อใช้งานเว็บเดโม่

**ไฟล์ที่ต้องมี**
- `Gradio/best_lraspp_mbv3.pth` — ไฟล์น้ำหนักโมเดล
- `Gradio/_classes.csv` — รายชื่อคลาส (หนึ่งคลาสต่อหนึ่งบรรทัด, index ตรงกับ training)

---

## 🧠 Tips (Colab / T4 Friendly)

- ถ้าเจอ **CUDA OOM** (หน่วยความจำไม่พอ)
  - ลด `BATCH`, ลด `IMGSZ` (เช่น 512/480), เพิ่ม `ACCUM` (gradient accumulation)
  - เปิด `AMP=1` (Automatic Mixed Precision)
  - ลด `NUM_WORKERS` หรือปิด `PRELOAD`
- เทรนเร็วขึ้น:
  - ใช้ `PRELOAD=1` สำหรับชุดข้อมูลเล็ก (แต่อาจใช้ RAM มากขึ้น)
  - ปรับ `FREEZE_BACKBONE_EPOCHS` > 0 สำหรับ Warm‑up
- ถ้า RAM สูงขึ้นเรื่อย ๆ ตรวจว่าไม่มีการสะสม tensor ในลูปเทรน (ensure `optimizer.zero_grad(set_to_none=True)` และ `with torch.no_grad()` ที่ inference)

---

## 📚 How it works (high‑level)

- **`main.py`** อ่านค่า config จาก `.env` → ดึง dataset ผ่าน Roboflow API → เตรียม DataLoader → สร้างโมเดล (เช่น `LRASPP MobileNetV3` / `DeepLabV3‑ResNet50`) → เทรนและบันทึก weights
- **`Gradio/app_semseg.py`** โหลด weights (`best_lraspp_mbv3.pth`) และไฟล์คลาส → ให้ผู้ใช้เลือกภาพ/วิดีโอ → แสดงผล segmentation พร้อมสรุปเบื้องต้น

---

## 🗂️ Dataset (Roboflow)

- โปรดตรวจสอบสิทธิ์การเข้าถึงโปรเจกต์ Roboflow และโควตาการดาวน์โหลด
- หากเปลี่ยน dataset ให้แก้ค่าที่ `.env` (`RF_WORKSPACE`, `RF_PROJECT`, `RF_VERSION`)

---

## 🧪 Reproducibility

- ระบุ seed/เวอร์ชันไลบรารีใน issue หรือ PR เพื่อช่วยกันดีบัก
- แนะนำให้ใส่ไฟล์ `requirements.txt` (ออปชัน) ถ้าต้องการ lock เวอร์ชัน

ตัวอย่าง `requirements.txt` แบบหลวม:
```
torch
torchvision
roboflow
python-dotenv
tqdm
gradio
pillow
opencv-python
```

---

## 🤝 Contributing

ยินดีรับ PR/Issue:
- เพิ่มสถาปัตยกรรมใหม่
- ปรับปรุงประสิทธิภาพบน Colab/T4
- เพิ่มยูทิลิตี้สำหรับ evaluation (mIoU, Pixel Acc, Confusion Matrix)

---

## 📄 License

โปรดระบุไลเซนส์ที่ต้องการใช้งาน (เช่น MIT, Apache‑2.0) ในไฟล์ `LICENSE` ของโปรเจกต์

---

## ✨ Acknowledgements

- [Roboflow](https://roboflow.com/) สำหรับ dataset pipeline
- PyTorch / TorchVision
- Gradio สำหรับ UI แบบเร็วและเรียบง่าย

---

> หากต้องการ README ภาษาอังกฤษล้วน ๆ หรือเพิ่ม badge/ภาพตัวอย่าง/ผลเทรน บอกได้เลยครับ
