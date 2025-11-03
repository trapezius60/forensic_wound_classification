import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import time
import requests
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ------------------- Page Config -------------------
st.set_page_config(page_title="Wound Detection App", page_icon="🤕", layout="wide")

# ------------------- Header -------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🤕 Forensic Wound Detection 🔎</h1>
    """,
    unsafe_allow_html=True
)
st.write("Upload an image or use your webcam for live detection.")

# ------------------- Wound Descriptions -------------------
wound_descriptions = {
    "wound_hesitation": "บาดแผลลังเล (Hesitation wound): มักพบในผู้พยายามทำร้ายตนเอง...",
    "wound_laceration": "บาดแผลฉีกขาดขอบไม่เรียบ (Laceration): เกิดจากวัตถุแข็งไม่มีคมกระแทก...",
    "wound_open_fracture": "บาดแผลกระดูกหักแบบเปิด (open fracture): เกิดจากกระดูกหักทิ่มออกมานอกผิวหนัง...",
    "wound_burn": "บาดแผลไหม้ (burn): ให้ดูความลึกและการกระจายของบาดแผล...",
    "wound_hanging": "บาดแผลกดรัดบริเวณลำคอ แขวนคอ (hanging): โดยทั่วไปหากพบลักษณะการกดรัดเฉียงขึ้น...",
    "wound_strangulation": "บาดแผลกดรัดบริเวณลำคอ รัดคอ (strangulation): มีลักษณะการกดรัดแนวขวาง...",
    "gsw_entrance": "บาดแผลทางเข้ากระสุนปืน (gunshot wound entrance): ลักษณะ punch-out lesion...",
    "gsw_exit": "บาดแผลทางออกกระสุนปืน (gunshot wound exit): โดยทั่วไปจะขนาดใหญ่กว่าทางเข้า..."
}

# ------------------- Model Download & Cache -------------------
@st.cache_resource
def get_model_path():
    """Download best.pt from GitHub once and reuse locally"""
    model_url = "https://raw.githubusercontent.com/trapezius60/forensic_wound_classification/main/models/best.pt"
    os.makedirs("models", exist_ok=True)
    local_path = "models/best.pt"

    if not os.path.exists(local_path):
        with st.spinner("Downloading YOLO model from GitHub..."):
            r = requests.get(model_url)
            if r.status_code != 200:
                st.error("Failed to download model file from GitHub.")
                st.stop()
            with open(local_path, "wb") as f:
                f.write(r.content)
    return local_path


@st.cache_resource
def load_model_with_retry(model_path, retries=3, delay=5):
    """Safely import cv2 & YOLO after environment wakeup"""
    for i in range(retries):
        try:
            import cv2
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model
        except Exception as e:
            st.warning(f"Model load failed (attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
    st.error("Failed to load YOLO model after multiple attempts.")
    st.stop()


model_path = get_model_path()
model = load_model_with_retry(model_path)

# ------------------- Confidence Slider -------------------
conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    import cv2  # delayed import (avoids cold boot crash)
    img = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run detection
    results = model(img_cv, conf=conf_thresh)
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Display annotated image
    st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

    # Extract detected wound types
    detected_classes = {results[0].names[int(c)] for c in results[0].boxes.cls.cpu().numpy()}

    # Show descriptions
    if detected_classes:
        st.subheader("📝 Wound Type Descriptions")
        desc_texts = [
            f"**{cls}**: {wound_descriptions.get(cls, '(No description available)')}"
            for cls in detected_classes
        ]
        st.info("\n\n".join(desc_texts))

    # Save for download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, annotated_bgr)
    st.download_button("Download Annotated Image", data=open(temp_file.name, "rb").read(), file_name="detection.png")

# ------------------- Webcam Live Detection -------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_frame = None

    def transform(self, frame):
        import cv2
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=conf_thresh)
        annotated = results[0].plot()
        self.captured_frame = annotated
        return annotated


webrtc_ctx = webrtc_streamer(
    key="wound-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_transform=True,
)

# ------------------- Capture Button -------------------
st.markdown("---")
if webrtc_ctx.video_transformer:
    if st.button("📸 Capture & Download Current Frame"):
        frame_bgr = webrtc_ctx.video_transformer.captured_frame
        if frame_bgr is not None:
            import cv2
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            st.download_button("Download Captured Image", data=open(temp_file.name, "rb").read(), file_name="capture.png")
        else:
            st.warning("No frame captured yet! Please wait for the webcam to initialize.")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; font-size:14px; color:gray;'>
        Forensic Education Version: 1.2.0 | © 2025 BH <br>
        <div>
            <a href="https://docs.google.com/document/d/18KlYv7Xbp3Y4Snatfez_jff0OW7DWKPoYP3HA3fx2cQ/edit?usp=sharing" target="_blank">📄 User Manual</a> |
            <a href="https://forms.gle/WgGnkcUQPafyhmng8" target="_blank">👍 Feedback Please</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
