# Real-Time Video Anomaly Detection for Bank Locker Security

This project implements a **Real-Time Video Anomaly Detection System for Bank Locker Security** using **Deep Learning and Computer Vision techniques**.
The system automatically detects suspicious activities in surveillance footage and alerts security personnel through alarms, snapshots, and SMS notifications.

---

## Project Overview

Traditional CCTV surveillance systems rely heavily on human monitoring, which can lead to delayed response or missed suspicious activities. This project introduces an intelligent surveillance system that can automatically analyze video streams and detect abnormal behavior in real-time.

The system uses a **Deep Learning Autoencoder model** trained on normal video frames to identify anomalies based on reconstruction error. Additionally, **YOLOv8 object detection** is integrated to detect dangerous objects such as knives, guns, and scissors.

To further enhance system reliability, **motion detection** and **camera tamper detection** techniques are incorporated. When suspicious activity is detected continuously, the system generates an **alarm sound, captures a snapshot, and sends an SMS alert using Twilio API**.

---

## Key Features

* Real-time video surveillance monitoring
* Deep learning based anomaly detection using Autoencoder
* Weapon detection using YOLOv8
* Motion detection using frame differencing
* Camera tamper detection using brightness and blur analysis
* Alarm system using beep sound
* Automatic snapshot capture during anomalies
* SMS alert system using Twilio API
* Hybrid anomaly scoring system to reduce false alarms

---

## Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* YOLOv8 (Ultralytics)
* NumPy
* Twilio API

---

## System Architecture

Camera Input
↓
Frame Capture using OpenCV
↓
Image Preprocessing (Resize & Grayscale)
↓
Autoencoder Reconstruction Error Calculation
↓
YOLOv8 Object Detection
↓
Motion Detection
↓
Camera Tamper Detection
↓
Hybrid Anomaly Scoring
↓
Alert Generation (Beep + Snapshot + SMS)

---

## Project Structure

Video-Anomaly-Detection

├── bank_security.py
├── calibrate.py
├── train_model.py
├── video_to_frames.py
├── yolo_detect.py
├── twilio_config.py

├── model/
│   └── anomaly_model.h5

├── dataset/
│   └── README.txt

├── snapshots/

├── README.md
├── requirements.txt
├── report.pdf
└── ppt.pptx

---

## Dataset

This project uses the **CUHK Avenue Dataset**, which is widely used for anomaly detection research.

Dataset Link:
https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

Dataset preparation process:

1. Surveillance videos are downloaded from the dataset.
2. Videos are converted into frames using `video_to_frames.py`.
3. Frames containing normal activities are used for training the autoencoder model.
4. The trained model learns normal patterns and detects abnormal frames using reconstruction error.

Due to GitHub file size limitations, only a small sample dataset is included in the repository.

---

## Installation

Clone the repository:

git clone https://github.com/praxmadhan/Video-Anomaly-Detection.git

Navigate to the project folder:

cd video-anomaly-detection

Install required libraries:

pip install -r requirements.txt

Run the system:

python bank_security.py

---

## Results

The system successfully detects various abnormal scenarios including:

* Suspicious motion
* Weapon presence
* Camera blocking or tampering
* Unusual activities

When an anomaly is detected:

1. Alarm sound is triggered.
2. Snapshot of the frame is saved.
3. SMS alert is sent to the registered phone number.

---

## Future Enhancements

* GPU acceleration for faster processing
* Multi-camera surveillance system
* Cloud-based monitoring dashboard
* Face recognition integration
* AI-based behavioral analysis

---

## Author

Prasanna N
B.Tech Information Technology
Bharath Niketan Engineering College
Anna University

---

## License

This project is developed for academic purposes as part of the Final Year B.Tech Project.
