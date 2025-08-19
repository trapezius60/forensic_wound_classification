# 🤕 forensic wound classification

A **real-time wound detection web app** built with [Streamlit](https://streamlit.io/) and [YOLOv8](https://github.com/ultralytics/ultralytics).  
The app allows you to detect wounds on images, uploaded files, or directly from a webcam with live streaming.
for medical education during learning simulate scene investigation.
---

## 🚀 Features

- 📷 **Upload Image** – Detect wounds in uploaded images.  
- 🎥 **Live Camera** – Perform real-time wound detection via webcam.  
- 📸 **Take a Snapshot** – Capture an image from the live feed and run detection.  
- 💾 **Download Results** – Save processed images with bounding boxes.  
- 📊 **Post-Evaluation** – Evaluate detection results after processing.  

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/trapezius60/forensic_wound_classification.git
cd forensic_wound_classification


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

Install dependencies:

pip install -r requirements.txt
---

## ▶️ Usage

Run the Streamlit app:
streamlit run app.py

Open your browser at http://localhost:8501.

##📂 Project Structure
forensic_wound_calssification/
│── app.py # Main Streamlit application
│── models/
│ └── yolov8n.pt # YOLOv8 model weights
│── requirements.txt # Python dependencies
│── README.md # Project documentation

##📦 Requirements

streamlit>=1.30.0
ultralytics==8.3.0        # works with Python 3.13 on Streamlit Cloud
opencv-python-headless==4.8.1.78
streamlit-webrtc==0.63.0
numpy>=1.26.0
Pillow>=10.0.0

Install everything via:
pip install -r requirements.txt

##🧠 Model

The app uses YOLOv8
 for wound detection.
You can replace yolov8n.pt with your custom-trained model to detect specific wound types.

📸 Screenshots
Live Detection
Uploaded Image

##📌 Roadmap

 Add wound classification (type/severity).
 Support video file uploads.
 Enhance post-class evaluation metrics.
 Deploy to cloud (Streamlit Cloud / Hugging Face Spaces).

##🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

##📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

##🙏 Acknowledgments

Ultralytics YOLOv8
Streamlit
OpenCV
