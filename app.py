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

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ------------------- Confidence Slider -------------------
conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ------------------- Image Upload -------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # convert for YOLO input
    
    # Small sleep for stability
    time.sleep(0.1)

    results = model(img_cv, conf=conf_thresh)
    annotated_rgb = results[0].plot()  # RGB for display

    # Display annotated image
    st.image(
        annotated_rgb,
        caption="Detection Result",
        use_container_width=True
    )

    # Save for download (convert RGB -> BGR)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, annotated_bgr)
    st.download_button(
        "Download Annotated Image",
        data=open(temp_file.name, "rb").read(),
        file_name="detection.png"
    )

# ------------------- Webcam Live Detection -------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_frame = None  # store RGB for display/capture

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        time.sleep(0.01)  # tiny sleep for smoother frames

        results = model(img, conf=conf_thresh)
        annotated_rgb = results[0].plot()  # RGB for WebRTC preview
        self.captured_frame = annotated_rgb  # store RGB for capture/download
        return annotated_rgb

# ------------------- Initialize Webcam -------------------
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
        frame_rgb = webrtc_ctx.video_transformer.captured_frame
        if frame_rgb is not None:
            # Convert RGB -> BGR for saving
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, frame_bgr)

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
Version: 1.0.0 | Updated: August 2025 | Powered by BH <br>
<div>
  <a href="https://docs.google.com/document/d/18KlYv7Xbp3Y4Snatfez_jff0OW7DWKPoYP3HA3fx2cQ/edit?usp=sharing" target="_blank">📄 User Manual</a> | 
  <a href="https://forms.gle/WgGnkcUQPafyhmng8" target="_blank">👍 Feedback Please</a>
</div>
""", unsafe_allow_html=True)
