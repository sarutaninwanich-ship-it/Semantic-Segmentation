# ==============================
# 🔹 STEP 3 — Clone เฉพาะโฟลเดอร์ Gradio จาก GitHub (Sparse Checkout)
# ==============================

# เริ่มที่ /content
%cd /content
!rm -rf Semantic-Segmentation

# 1) สร้าง repo ว่าง
!git init Semantic-Segmentation
%cd /content/Semantic-Segmentation

# 2) ตั้งค่า remote และเปิดโหมด sparse checkout
!git remote add origin https://github.com/sarutaninwanich-ship-it/Semantic-Segmentation.git
!git config core.sparseCheckout true

# 3) ระบุว่าเอาเฉพาะโฟลเดอร์ Gradio
!echo "Gradio/" >> .git/info/sparse-checkout

# 4) ดึงข้อมูลเฉพาะโฟลเดอร์นั้นจาก branch main
!git pull origin main

# 5) เข้าไปในโฟลเดอร์ Gradio และตรวจไฟล์
%cd /content/Semantic-Segmentation/Gradio
!ls -la

# 6) ตรวจสอบว่ามีไฟล์สำคัญครบไหม
import os
need = ["app_semseg.py", "_classes.csv", "best_lraspp_mbv3.pth"]
for n in need:
    print(f"{n} → {'✅' if os.path.exists(n) else '❌ MISSING'}")

# 7) ติดตั้งไลบรารีจำเป็น
%pip install -q torch torchvision roboflow python-dotenv tqdm gradio opencv-python pillow numpy matplotlib

# 8) รันแอป Gradio
!python app_semseg.py --share
