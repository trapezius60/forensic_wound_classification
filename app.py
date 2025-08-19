import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ----------------------
# Load YOLO model
# ----------------------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("models/best.pt")

model = load_model()

# ----------------------
# OpenCV import
# ----------------------
try:
    import cv2
except ImportError:
    st.error("⚠️ OpenCV failed to load. Please refresh after a few seconds.")

# ----------------------
# App Configuration
# ----------------------
st.set_page_config(
    page_title="Wound Detection for Forensic Simulation",
    page_icon="🤕",
    layout="wide"
)

# ----------------------
# Header Section
# ----------------------
st.title("🤕 Wound Detection App 🔎")
st.markdown("Forensic Medicine Student Simulation & Learning")

# ----------------------
# Sidebar Resources
# ----------------------
st.sidebar.header("Resources")
st.sidebar.markdown(
    """
    <a href='https://docs.google.com/document/d/18KlYv7Xbp3Y4Snatfez_jff0OW7DWKPoYP3HA3fx2cQ/edit?usp=sharing' target='_blank'>📄 User Manual</a><br>
    <a href='https://forms.gle/WgGnkcUQPafyhmng8' target='_blank'>👍 Feedback Form</a>
    """,
    unsafe_allow_html=True
)

st.write("Upload an image or use your webcam for live detection")

# ----------------------
# Confidence Slider
# ----------------------
conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# ----------------------
# Upload Section
# ----------------------
st.subheader("📤 Upload Wound Image")
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(img_array, caption="Uploaded Image", use_container_width=True)
    st.success("✅ Image uploaded successfully!")

    # Run detection
    results = model.predict(img_array, conf=conf_thresh)
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detection Result", use_container_width=True)

    # Download annotated image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, annotated_frame)
    st.download_button(
        "Download Annotated Image",
        data=open(temp_file.name, "rb").read(),
        file_name="detection.png"
    )

# ----------------------
# Webcam Live Detection
# ----------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.captured_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=conf_thresh)
        annotated = results[0].plot()
        self.captured_frame = annotated
        return annotated

webrtc_ctx = webrtc_streamer(
    key="wound-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_transform=True,
)

# ----------------------
# Capture Button
# ----------------------
st.markdown("---")
if webrtc_ctx.video_transformer:
    if st.button("📸 Capture & Download Current Frame"):
        frame = webrtc_ctx.video_transformer.captured_frame
        if frame is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, frame)
            st.download_button(
                "Download Captured Image",
                data=open(temp_file.name, "rb").read(),
                file_name="capture.png"
            )
        else:
            st.warning("No frame captured yet!")

# ----------------------
# Footer Section
# ----------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        "<a href='https://docs.google.com/document/d/18KlYv7Xbp3Y4Snatfez_jff0OW7DWKPoYP3HA3fx2cQ/edit?usp=sharing' target='_blank'>📄 User Manual</a>",
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        "<a href='https://forms.gle/WgGnkcUQPafyhmng8' target='_blank'>👍 Feedback Form</a>",
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        "<a href='https://forms.gle/your-post-class-form-link' target='_blank'>📝 Post-class Evaluation</a>",
        unsafe_allow_html=True
    )

# Footer Note
st.markdown(
    "<div style='text-align: center; font-size: 0.9em; color: gray;'>© 2025 Forensic Medicine Teaching App | Maharat Nakhon Ratchasima Hospital</div>",
    unsafe_allow_html=True
)

