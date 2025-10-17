# app_semseg_video.py
import os, cv2, time, tempfile, torch, gradio as gr
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIG ----------------
CKPT_PATH  = "best_lraspp_mbv3.pth"   # เปลี่ยนเป็นพาธ checkpoint ของอาจารย์
CLASS_FILE = "_classes.csv"           # ไฟล์รายชื่อคลาสแบบ CSV: Pixel Value, Class
IMGSZ      = 640
ALPHA      = 0.45
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_NUM_CLASSES = None  # ถ้าต้องการบังคับจำนวนคลาส เช่น 63 ให้ใส่เป็น int
MIN_PIXELS_FOR_LABEL = 120  # ป้องกัน noise; ต้องมีพิกเซลถึงกำหนดจึงจะวางป้ายชื่อ
# ----------------------------------------


# ---------- Utils: ckpt & classes ----------
def _read_ckpt_state(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ไม่พบ checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("checkpoint ต้องเป็น dict")
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd

def _num_classes_from_state(sd):
    for k in ("classifier.low_classifier.bias",
              "classifier.low_classifier.weight",
              "classifier.high_classifier.bias",
              "classifier.high_classifier.weight"):
        if k in sd:
            return sd[k].shape[0]
    raise RuntimeError("ไม่พบพารามิเตอร์บอกจำนวนคลาสใน checkpoint")

def load_classes(csv_path, require_n=None):
    names = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.lower().startswith("pixel"):  # header: Pixel Value, Class
                    continue
                parts = [p.strip() for p in s.split(",")]
                names.append(parts[-1])
    else:
        print("[WARN] ไม่พบไฟล์คลาส → ใช้ค่าเริ่มต้น ['background','object']")
        names = ["background", "object"]

    if require_n is not None:
        if len(names) < require_n:
            print(f"[WARN] เพิ่ม placeholder ให้ครบ {require_n}")
            names += [f"class_{i}" for i in range(len(names), require_n)]
        elif len(names) > require_n:
            print(f"[WARN] ตัดรายชื่อคลาสให้เหลือ {require_n}")
            names = names[:require_n]
    return names

def build_palette(n):
    rng = np.random.default_rng(2025)
    palette = [(60,60,60)]  # background = เทาเข้ม
    for _ in range(max(0, n-1)):
        palette.append(tuple(int(x) for x in rng.integers(0,256,3)))
    return palette

def load_model_and_metadata(ckpt_path, class_file, force_n=None):
    sd = _read_ckpt_state(ckpt_path)
    ckpt_n = force_n or _num_classes_from_state(sd)
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=ckpt_n)
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    class_names = load_classes(class_file, require_n=ckpt_n)
    palette = build_palette(ckpt_n)
    return model, class_names, palette, ckpt_n

MODEL, CLASS_NAMES, PALETTE, NUM_CLASSES = load_model_and_metadata(
    CKPT_PATH, CLASS_FILE, force_n=FORCE_NUM_CLASSES
)


# ---------- Preprocess / color ----------
def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    return t(img_rgb).unsqueeze(0)

def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    for i,c in enumerate(PALETTE):
        color[mask==i] = c
    return color


# ---------- Label drawing (skip 'background') ----------
def _load_font(size_px):
    for p in [
        "C:/Windows/Fonts/THSarabunNew.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size_px)
            except Exception:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size_px)
    except Exception:
        return ImageFont.load_default()

def add_class_labels(mask_color, pred_mask, class_names):
    """วาดชื่อคลาส (ยกเว้น background) พร้อมกล่องพื้นหลังดำโปร่งใส"""
    img_pil = Image.fromarray(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    H, W = pred_mask.shape
    font = _load_font(max(16, min(H, W)//45))

    for cls_id, name in enumerate(class_names):
        if cls_id == 0:  # ไม่วาด background
            continue
        ys, xs = np.where(pred_mask == cls_id)
        if ys.size < MIN_PIXELS_FOR_LABEL:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        text = name
        # ใช้ textbbox เพื่อคำนวณขนาดสี่เหลี่ยม
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x1, y1 = max(0, cx - tw//2 - 5), max(0, cy - th//2 - 3)
        x2, y2 = min(W, x1 + tw + 10), min(H, y1 + th + 6)
        draw.rectangle([(x1, y1), (x2, y2)], fill=(0,0,0,180))
        draw.text(((x1+x2)//2, (y1+y2)//2), text, font=font, fill=(255,255,255,255), anchor="mm")

    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# ---------- Core inference: image ----------
@torch.no_grad()
def predict_image(pil_image, alpha=ALPHA, draw_labels=True):
    if pil_image is None:
        return None, None
    img_bgr = np.array(pil_image)[..., ::-1]
    h, w = img_bgr.shape[:2]
    x = preprocess(cv2.resize(img_bgr, (IMGSZ, IMGSZ))).to(DEVICE)
    out = MODEL(x)["out"]
    out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)

    mask_color = colorize_mask(pred)
    if draw_labels:
        mask_color = add_class_labels(mask_color, pred, CLASS_NAMES)
    overlay = cv2.addWeighted(img_bgr, 1.0, mask_color, alpha, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)


# ---------- Core inference: video (สามโหมด) ----------
@torch.no_grad()
def predict_video(video, alpha=ALPHA, stride=1, draw_labels=False, out_mode="overlay", progress=gr.Progress()):
    """
    อ่านวิดีโอ → ทำนายทีละเฟรม → คืนพาธไฟล์ MP4
    out_mode: overlay / mask / side
    """
    # แปลง input ของ Gradio ให้เป็น path จริง
    vpath = None
    if isinstance(video, str) and os.path.exists(video):
        vpath = video
    elif isinstance(video, dict):
        for k in ("path", "name", "tempfile", "file"):
            p = video.get(k)
            if isinstance(p, str) and os.path.exists(p):
                vpath = p
                break
    if not vpath:
        return None

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

    stride = max(1, int(stride))
    fps_out = max(1.0, fps / stride)

    # lazy writer: จะเปิดเมื่อรู้ขนาดเฟรมจริง
    tmpdir = tempfile.mkdtemp(prefix="semseg_")
    out_path = os.path.join(tmpdir, f"seg_out_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = None

    frame_idx = 0
    total_iter = N if N > 0 else 1000000
    for _ in progress.tqdm(range(total_iter), desc="วิเคราะห์วิดีโอ"):
        ret, frame = cap.read()
        if not ret:
            break

        if stride > 1 and (frame_idx % stride != 0):
            frame_idx += 1
            continue

        # Inference เฟรมเดียว
        h, w = frame.shape[:2]
        x = preprocess(cv2.resize(frame, (IMGSZ, IMGSZ))).to(DEVICE)
        out = MODEL(x)["out"]
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)

        mask_color = colorize_mask(pred)
        if draw_labels:
            mask_color = add_class_labels(mask_color, pred, CLASS_NAMES)

        if out_mode == "overlay":
            out_frame = cv2.addWeighted(frame, 1.0, mask_color, alpha, 0)
        elif out_mode == "mask":
            out_frame = mask_color
        else:  # side-by-side
            if mask_color.shape[:2] != frame.shape[:2]:
                mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)
            out_frame = cv2.hconcat([frame, mask_color])

        # บังคับเฟรมเป็นเลขคู่ (บาง codec ต้องการ)
        oh, ow = out_frame.shape[:2]
        ow2 = ow - (ow % 2)
        oh2 = oh - (oh % 2)
        if ow2 != ow or oh2 != oh:
            out_frame = cv2.resize(out_frame, (ow2, oh2), interpolation=cv2.INTER_AREA)

        if vw is None:
            vw = cv2.VideoWriter(out_path, fourcc, fps_out, (out_frame.shape[1], out_frame.shape[0]))

        vw.write(out_frame)
        frame_idx += 1

        # safety stop สำหรับสตรีมไม่ทราบจำนวนเฟรม
        if N == 0 and frame_idx > 2000:
            break

    cap.release()
    if vw: vw.release()
    return out_path


# ---------- Gradio UI ----------
with gr.Blocks(title="Semantic Segmentation (Image + Video)") as demo:
    gr.Markdown("## 🧠 Semantic Segmentation — LRASPP MobileNetV3\nอัปโหลดภาพหรือวิดีโอเพื่อทดสอบการแบ่งส่วนเชิงความหมาย")

    # ---------- Tab: Image ----------
    with gr.Tab("ภาพนิ่ง"):
        with gr.Row():
            inp_img = gr.Image(type="pil", label="อัปโหลดภาพ (JPG/PNG)")
            with gr.Column():
                out_overlay = gr.Image(type="numpy", label="ซ้อนทับผลลัพธ์")
                out_mask    = gr.Image(type="numpy", label="Mask + ป้ายชื่อ")

        with gr.Row():
            alpha_img = gr.Slider(0.0, 1.0, value=ALPHA, step=0.05, label="ความทึบ Overlay")
            draw_lbl  = gr.Checkbox(value=True, label="แปะชื่อคลาสบน Mask (ยกเว้น background)")

        btn_img = gr.Button("ทำนายภาพ")
        btn_img.click(fn=predict_image, inputs=[inp_img, alpha_img, draw_lbl], outputs=[out_overlay, out_mask])

    # ---------- Tab: Video ----------
    with gr.Tab("วิดีโอ"):
        with gr.Row():
            inp_vid = gr.Video(label="อัปโหลดวิดีโอ", sources=["upload"], interactive=True)
            out_vid = gr.Video(label="วิดีโอผลลัพธ์ (MP4)", autoplay=True, show_download_button=True)

        with gr.Row():
            alpha_vid  = gr.Slider(0.0, 1.0, value=ALPHA, step=0.05, label="ความทึบ Overlay")
            stride_vid = gr.Slider(1, 8, value=1, step=1, label="ข้ามเฟรม (ยิ่งมากยิ่งเร็ว)")
            draw_lbl_v = gr.Checkbox(value=False, label="แปะชื่อคลาสบนเฟรม (ช้าลงเล็กน้อย)")
            out_mode_v = gr.Radio(
                choices=[("Overlay","overlay"), ("Mask-only","mask"), ("Side-by-side","side")],
                value="overlay",
                label="โหมดเอาต์พุตวิดีโอ"
            )

        btn_vid = gr.Button("ทำนายทั้งวิดีโอ")
        btn_vid.click(
            fn=predict_video,
            inputs=[inp_vid, alpha_vid, stride_vid, draw_lbl_v, out_mode_v],
            outputs=[out_vid],
            queue=True
        )

    gr.Markdown("### 🏷️ รายชื่อคลาส (ตาม checkpoint)")
    gr.Dataframe(headers=["ID","Class Name"], value=[[i,n] for i,n in enumerate(CLASS_NAMES)], interactive=False)

    with gr.Accordion("ℹ️ ข้อมูลระบบ", open=False):
        gr.Markdown(f"- **Device**: `{DEVICE}`\n- **Classes (from ckpt)**: `{NUM_CLASSES}`")

# เปิด share=True สำหรับ Colab/ภายนอก
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
