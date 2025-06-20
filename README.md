# 🎭 Face Emotion Recognition with Live Camera

A machine learning project that detects **human emotions** in real-time using your webcam. Built with a custom-trained **Convolutional Neural Network (CNN)** using TensorFlow and OpenCV.

---

## 📸 What It Does

- Detects faces using your webcam
- Classifies facial emotion: **happy, sad, angry, surprised, neutral**, etc.
- Trained on grayscale 48x48 facial image data
- Displays detected emotion label on-screen

---





## 🚀 How to Run

1. Clone this repo

```bash
git clone https://github.com/logyfarg/face-emotion-detector.git
cd face-emotion-detector

2. Install requirements:
pip install -r requirements.txt

3. Run the webcam:
python live_emotion_detector.py

🏗 Project Structure
face-emotion-detector/
│
├── model/
│   └── emotion_model.h5         # Trained model file
├── train_model.py               # CNN training script
├── live_emotion_detector.py     # Webcam prediction code
├── requirements.txt             # Python dependencies
└── README.md                    # This file

Tech Stack
Python

TensorFlow / Keras

OpenCV

NumPy
👩‍💻 Author
Built with ❤️ by Logina Mahmoud

