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


#🧬 Forensic Wound Classification — Label Map & Data Dictionary

This project uses a unified label map across all dataset annotations, YOLOv8 configuration files, training scripts, and publication figures.
All labels are case-sensitive and written in lowercase with underscores (_) separating words.

🗂️ Label Map
ID	Class Name	Description	Example Appearance
0	gsw_entrance	Gunshot wound (entrance) — small circular or oval wound with inward beveling	Typically found at bullet entry site
1	gsw_exit	Gunshot wound (exit) — irregular laceration with outward beveling	Typically larger than entrance wound
2	wound_burn	Thermal injury with charring or blistering of skin	May appear dark brown to black with irregular margins
3	wound_hanging	Ligature mark or neck indentation consistent with hanging	Typically horizontal or oblique mark on neck
4	wound_hesitation	Superficial, parallel cuts consistent with self-inflicted hesitation marks	Often found near fatal incised wound sites
5	wound_laceration	Tear in skin due to blunt force trauma	Irregular edges with tissue bridging
6	wound_open_fracture	Fracture with exposed bone or tissue disruption	Commonly associated with high-impact trauma
7	wound_strangulation	Neck compression marks with petechiae or abrasions	May show patterned bruising or linear marks
📦 Dataset Summary

Total Images: 595

Total Annotated Instances: 569

Train/Validation/Test Split: 70% / 20% / 10%

Image Resolution: 640 × 640 px

Preprocessing: Grayscale conversion (CRT phosphor filter), stretch mode resizing

⚙️ Model Configuration

Architecture: YOLOv8n (Ultralytics v8.3.191)

Backbone: Pretrained COCO weights (transfer learning)

Optimizer: AdamW

Learning Rate: 0.01

Epochs: 100 (early stopping enabled)

Confidence Threshold: 0.25 (default)

NMS IoU Threshold: 0.7 (default)

📊 Evaluation Metrics
Metric	Description
Precision (P)	Proportion of predicted positives that are correct
Recall (R)	Proportion of actual positives correctly detected
mAP@50 / mAP@50–95	Mean average precision at IoU thresholds
AP@75	Average precision at IoU 0.75 (stricter localization)
ECE (Expected Calibration Error)	Measures how well confidence scores match true accuracy
🔗 Reproducibility Notes

Weights: best.pt (6 MB) — available in repository

Configuration: data.yaml and model.yaml provided

Environment:

Python 3.12

Ultralytics 8.3.191

PyTorch 2.8.0 + CUDA 12.6

GPU Used: NVIDIA Tesla T4 (Google Colab)

🧠 Citation

If you use this dataset or model, please cite:

Hanterdsith, B. Integrating Forensic Wound Detection and Classification with Custom Vision in Simulation-Based Teaching: An Educational Innovation for Medical Students. (2025)
