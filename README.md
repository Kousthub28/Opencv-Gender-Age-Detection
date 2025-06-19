# 🤖 Gender and Age Detection using OpenCV, Flask & Voice

This project performs **real-time gender and age detection** from a webcam using **OpenCV**, **Flask**, and **deep learning models**. It also plays a **custom audio greeting** based on the detected gender.

---

## 🚀 Features

- 🎥 Real-time webcam face detection
- 🧠 Deep learning-based gender & age prediction (Caffe models)
- 🔊 Voice output using your own `.wav` audio
- 🌐 Flask front-end served via Ngrok for public access
- 👨‍💻 Clean folder structure for easy deployment

---

## 📁 Folder Structure

Opencv-Gender-Age-Detection/
├── app.py # Flask main app
├── gender_voice.py # (optional) script version without Flask
├── templates/
│ └── index.html # Front-end UI (basic)
├── gender_net.caffemodel # Gender prediction model
├── deploy_gender.prototxt # Gender model config
├── age_net.caffemodel # Age prediction model
├── deploy_age.prototxt # Age model config
├── male_voice.wav # Custom greeting for male
├── female_voice.wav # Custom greeting for female
├── ngrok.exe # Ngrok for deployment (optional)
├── .gitattributes # Git LFS tracking
├── .gitignore # Ignore ngrok/model/audio files
└── README.md # You're reading it!

2. Install Dependencies
   pip install flask opencv-python simpleaudio
   pip install opencv-python opencv-python-headless numpy pyttsx3
4. Run the App
   python app.py
Go to:
http://127.0.0.1:5000/ in your browser
🔊 Custom Voice Support
Place two .wav files in your root folder:

male_voice.wav → Plays when a male is detected

female_voice.wav → Plays when a female is detected
