import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import time

# ------------------- Page Config -------------------
st.set_page_config(page_title="Wound Detection App", page_icon="🤕", layout="wide")

# ------------------- Header -------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🤕 Forensic Wound Detection 🔎</h1>
    """,
    unsafe_allow_html=True
)
st.write("Upload an image or use your webcam for live detection")

# ------------------- Wound Descriptions -------------------
wound_descriptions = {
    "wound_hesitation": "บาดแผลลังเล (Hesitation wound): มักพบในผู้พยายามทำร้ายตนเอง มีลักษณะเป็นแผลตื้นหลายแผล อยู่ใกล้แผลหลัก",
    "wound_laceration": "บาดแผลฉีกขาด (Laceration): เกิดจากแรงกระแทกหรือการฉีกขาดของผิวหนัง ขอบแผลไม่เรียบ",
    "wound_incision": "บาดแผลถูกฟัน/กรีด (Incised wound): ขอบแผลเรียบคม เกิดจากของมีคม",
    "wound_contusion": "บาดแผลฟกช้ำ (Contusion): เกิดจากแรงกระแทก ทำให้หลอดเลือดใต้ผิวหนังแตก",
    "wound_stab": "บาดแผลแทง (Stab wound): มีความลึกมากกว่าความยาว ปกติเกิดจากของมีคมปลายแหลม"
    # ➕ Add more classes if your model has them
}

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ------------------- Confidence Slider -------------------
conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg","png","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # BGR for YOLO

    # Run detection
    results = model(img_cv, conf=conf_thresh)
    annotated_bgr = results[0].plot()  # YOLO returns BGR
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB for display

    # Display annotated image
    st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

    # Extract detected wound types
    detected_classes = set()
    for r in results[0].boxes.cls.cpu().numpy():
        cls_name = results[0].names[int(r)]
        detected_classes.add(cls_name)

    # Show descriptions if available
    if detected_classes:
        st.subheader("📝 Wound Type Descriptions")
        desc_texts = []
        for cls in detected_classes:
            if cls in wound_descriptions:
                desc_texts.append(f"**{cls}**: {wound_descriptions[cls]}")
            else:
                desc_texts.append(f"**{cls}**: (No description available)")
        st.info("\n\n".join(desc_texts))

    # Save for download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, annotated_bgr)  # save BGR
    st.download_button(
        "Download Annotated Image",
        data=open(temp_file.name, "rb").read(),
        file_name="detection.png"
    )

# ------------------- Webcam Live Detection -------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_frame = None  # store BGR frame for capture/download

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # input BGR from webcam
        results = model(img, conf=conf_thresh)
        annotated = results[0].plot()  # BGR for WebRTC preview
        self.captured_frame = annotated  # store for capture/download
        return annotated  # BGR preview (WebRTC expects BGR)

# Initialize webcam
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
            # Convert BGR -> RGB -> BGR for saving (ensures correct color)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))  # save correctly
            st.download_button(
                "Download Captured Image",
                data=open(temp_file.name, "rb").read(),
                file_name="capture.png"
            )
        else:
            st.warning("No frame captured yet! Please wait for the webcam to initialize.")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; font-size:14px; color:gray;'>
Forensic education Version: 1.0.0 | © 2025 BH <br>
<div>
  <a href="https://docs.google.com/document/d/18KlYv7Xbp3Y4Snatfez_jff0OW7DWKPoYP3HA3fx2cQ/edit?usp=sharing" target="_blank">📄 User Manual</a> | 
  <a href="https://forms.gle/WgGnkcUQPafyhmng8" target="_blank">👍 Feedback Please</a>
</div>
""", unsafe_allow_html=True)
