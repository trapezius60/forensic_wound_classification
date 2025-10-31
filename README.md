# 🤕 Forensic Wound Classification

A **real-time wound detection web app** for medical education built with [Streamlit](https://streamlit.io/) and [YOLOv8](https://github.com/ultralytics/ultralytics). Designed for learning simulated scene investigation during forensic pathology training.

---

## 🚀 Features

- 📷 **Upload Image** – Detect wounds in uploaded images
- 🎥 **Live Camera** – Real-time wound detection via webcam
- 📸 **Snapshot Capture** – Capture and analyze images from live feed
- 💾 **Download Results** – Save processed images with bounding boxes
- 📊 **Post-Evaluation** – Evaluate detection results after processing

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/trapezius60/forensic_wound_classification.git
cd forensic_wound_classification

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📂 Project Structure

```
forensic_wound_classification/
├── app.py                 # Main Streamlit application
├── models/
│   └── best.pt            # YOLOv8 trained weights (6 MB)
├── requirements.txt       # Python dependencies
└── LICENSE                # MIT: Free to use, modify, and redistribute with attribution
└── README.md              # Documentation

```

---

## 📦 Requirements

```txt
streamlit>=1.30.0
ultralytics==8.3.0
opencv-python-headless==4.8.1.78
streamlit-webrtc==0.63.0
numpy>=1.26.0
Pillow>=10.0.0
```

---

## 🗂️ Wound Classification Labels

| ID | Class Name | Description |
|---|---|---|
| 0 | `gsw_entrance` | Gunshot wound entrance - small circular/oval with inward beveling |
| 1 | `gsw_exit` | Gunshot wound exit - irregular laceration with outward beveling |
| 2 | `wound_burn` | Thermal injury with charring or blistering |
| 3 | `wound_hanging` | Ligature mark on neck consistent with hanging |
| 4 | `wound_hesitation` | Superficial parallel cuts (self-inflicted hesitation marks) |
| 5 | `wound_laceration` | Blunt force tear with irregular edges and tissue bridging |
| 6 | `wound_open_fracture` | Fracture with exposed bone or tissue disruption |
| 7 | `wound_strangulation` | Neck compression marks with petechiae or abrasions |

---

## 🧠 Model Details

**Architecture:** YOLOv8n (Ultralytics v8.3.191)  
**Training Data:** 595 images, 569 annotated instances  
**Split:** 70% train / 20% validation / 10% test  
**Resolution:** 640 × 640 px (grayscale, CRT phosphor filter)  
**Optimizer:** AdamW (LR: 0.01, 100 epochs with early stopping)  
**Metrics:** Precision, Recall, mAP@50, mAP@50-95, ECE

**Environment:**
- Python 3.12
- Ultralytics 8.3.191
- PyTorch 2.8.0 + CUDA 12.6

---

## 📌 Roadmap

- [ ] Add severity grading and classification confidence scores
- [ ] Support video file uploads
- [ ] Enhanced evaluation metrics dashboard
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

This project — “Forensic Wound Detection and Classification with YOLOv8n:
An Educational Innovation for Medical Students” — is released under a dual license model to ensure open research access while maintaining responsible use.

#Component	License	Description
Source Code (Streamlit App, Python)	[MIT License](./LICENSE)
Educational and research use only; commercial use prohibited.
Trained Model (best.pt)	CC BY-NC 4.0
	May be reused for academic research or teaching with attribution.

---

## ⚖️ Ethical & Responsible Use Disclaimer

This software and dataset are intended solely for educational and research use in the context of medical and forensic science training.
They are not validated for clinical diagnosis, forensic adjudication, or legal proceedings.

All inferences produced by the model must be interpreted under human expert supervision.
The authors assume no liability for any misuse of this project beyond its intended educational scope.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

---

## 📖 Citation

If you use this dataset or model, please cite:

```
Hanterdsith, B. (2025). Integrating Forensic Wound Detection and Classification 
with Custom Vision in Simulation-Based Teaching: An Educational Innovation for 
Medical Students.
```
