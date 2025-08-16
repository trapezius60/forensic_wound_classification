import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ------------------- Page Config -------------------
st.set_page_config(page_title="Wound Detection App", layout="wide")

# ------------------- Header -------------------
st.title("🩹 Wound Detection YOLOv8")
st.write("Upload an image or use your webcam for live detection")

# ------------------- Load Model -------------------
# Update with your local model path
model = YOLO("models/best.pt")

# ------------------- Confidence Slider -------------------
conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model(img_cv, conf=conf_thresh)
    annotated_frame = results[0].plot()
    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
    
    # Download annotated image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, annotated_frame)
    st.download_button("Download Annotated Image", data=open(temp_file.name, "rb"), file_name="detection.png")

# ------------------- Webcam Live Detection -------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=conf_thresh)
        annotated = results[0].plot()
        self.captured_frame = annotated
        return annotated

# ------------------- Use back camera by default -------------------
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
        frame = webrtc_ctx.video_transformer.captured_frame
        if frame is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, frame)
            st.download_button("Download Captured Image", data=open(temp_file.name, "rb"), file_name="capture.png")
        else:
            st.warning("No frame captured yet!")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; font-size:14px; color:gray;'>
Version: 1.0.0 | Updated: August 2025 | Powered by BH <br>
<a href='https://docs.google.com/document/d/18KlYv7Xbp3Y4Snatfez_jff0OW7DWKPoYP3HA3fx2cQ/edit?usp=sharing' target='_blank'>📄 User Manual</a>
</div>
""", unsafe_allow_html=True)

