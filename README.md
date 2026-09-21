# 🎙️ Audio Vision: ESC-50 Environmental Sound Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B.svg)

## 📌 Overview
Proyek ini mengonversi masalah pemrosesan sinyal audio menjadi masalah *Computer Vision* menggunakan pendekatan *Transfer Learning*. Sistem ini mengekstraksi representasi spektral dari gelombang suara mentah menjadi **Mel-Spectrogram 3-Channel** (dinamika temporal *Delta* dan *Delta-Delta*), yang kemudian diproses menggunakan arsitektur **ResNet50**.

Model ini dilatih secara khusus menggunakan dataset ESC-50 untuk mengklasifikasikan 50 jenis suara lingkungan dunia nyata ke dalam 5 dimensi utama (Suara Alam, Manusia, Hewan, Eksterior/Urban, dan Interior/Domestik).

🔗 **[https://esc50-audioclassification.streamlit.app/]** *(-> Klik di sini untuk mencoba Live App/Demo)*

## 📊 Model Performance
* **Model Architecture:** ResNet50 (Pre-trained)
* **Test Accuracy:** 84.00%
* **Feature Extraction:** Librosa (Log-Mel Spectrogram + Delta + Delta-Delta)

## 🚀 How to Run Locally
Karena ukuran file model (`best_model.pth`) melebihi 100MB, file tersebut di-hosting secara terpisah di Google Drive. Sistem akan mengunduhnya secara otomatis saat pertama kali dijalankan menggunakan `gdown`.

1. Clone repositori ini:
   ```bash
   git clone [https://github.com/](https://github.com/)[username-github-kamu]/esc50-audio-classifier.git
