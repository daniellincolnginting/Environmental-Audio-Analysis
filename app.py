import streamlit as st
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision.transforms as tv_transforms
import librosa
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from torchvision import models
import gdown

# ====================================
# CONFIG & PAGE SETUP
# ====================================

st.set_page_config(
    page_title="Audio Classification | ESC-50",
    page_icon="🌊",
    layout="wide", 
    initial_sidebar_state="expanded"
)

SAMPLE_RATE = 44100
N_FFT       = 2048
HOP_LENGTH  = 512
N_MELS      = 128
TOP_DB      = 80

# ====================================
# LABELS & CATEGORIES
# ====================================

CLASS_NAMES = [
    'Dog', 'Rooster', 'Pig', 'Cow', 'Frog',
    'Cat', 'Hen', 'Insects', 'Sheep', 'Crow',
    'Rain', 'Sea waves', 'Crackling fire', 'Crickets', 'Chirping birds',
    'Water drops', 'Wind', 'Pouring water', 'Toilet flush', 'Thunderstorm',
    'Crying baby', 'Sneezing', 'Clapping', 'Breathing', 'Coughing',
    'Footsteps', 'Laughing', 'Brushing teeth', 'Snoring', 'Drinking - sipping',
    'Door knock', 'Mouse click', 'Keyboard typing', 'Door - wood creaks', 'Can opening',
    'Washing machine', 'Vacuum cleaner', 'Clock alarm', 'Clock tick', 'Glass breaking',
    'Helicopter', 'Chainsaw', 'Siren', 'Car horn', 'Engine',
    'Train', 'Church bells', 'Airplane', 'Fireworks', 'Hand saw'
]

CATEGORY_MAP = {
    'Dog': ('Animals', '🐾'), 'Rooster': ('Animals', '🐾'), 'Pig': ('Animals', '🐾'),
    'Cow': ('Animals', '🐾'), 'Frog': ('Animals', '🐾'), 'Cat': ('Animals', '🐾'),
    'Hen': ('Animals', '🐾'), 'Insects': ('Animals', '🐾'), 'Sheep': ('Animals', '🐾'),
    'Crow': ('Animals', '🐾'),
    'Rain': ('Natural Soundscapes', '🌿'), 'Sea waves': ('Natural Soundscapes', '🌿'),
    'Crackling fire': ('Natural Soundscapes', '🌿'), 'Crickets': ('Natural Soundscapes', '🌿'),
    'Chirping birds': ('Natural Soundscapes', '🌿'), 'Water drops': ('Natural Soundscapes', '🌿'),
    'Wind': ('Natural Soundscapes', '🌿'), 'Pouring water': ('Natural Soundscapes', '🌿'),
    'Toilet flush': ('Natural Soundscapes', '🌿'), 'Thunderstorm': ('Natural Soundscapes', '🌿'),
    'Crying baby': ('Human Sounds', '🗣️'), 'Sneezing': ('Human Sounds', '🗣️'),
    'Clapping': ('Human Sounds', '🗣️'), 'Breathing': ('Human Sounds', '🗣️'),
    'Coughing': ('Human Sounds', '🗣️'), 'Footsteps': ('Human Sounds', '🗣️'),
    'Laughing': ('Human Sounds', '🗣️'), 'Brushing teeth': ('Human Sounds', '🗣️'),
    'Snoring': ('Human Sounds', '🗣️'), 'Drinking - sipping': ('Human Sounds', '🗣️'),
    'Door knock': ('Interior & Domestic', '🏠'), 'Mouse click': ('Interior & Domestic', '🏠'),
    'Keyboard typing': ('Interior & Domestic', '🏠'), 'Door - wood creaks': ('Interior & Domestic', '🏠'),
    'Can opening': ('Interior & Domestic', '🏠'), 'Washing machine': ('Interior & Domestic', '🏠'),
    'Vacuum cleaner': ('Interior & Domestic', '🏠'), 'Clock alarm': ('Interior & Domestic', '🏠'),
    'Clock tick': ('Interior & Domestic', '🏠'), 'Glass breaking': ('Interior & Domestic', '🏠'),
    'Helicopter': ('Exterior & Urban', '🏙️'), 'Chainsaw': ('Exterior & Urban', '🏙️'),
    'Siren': ('Exterior & Urban', '🏙️'), 'Car horn': ('Exterior & Urban', '🏙️'),
    'Engine': ('Exterior & Urban', '🏙️'), 'Train': ('Exterior & Urban', '🏙️'),
    'Church bells': ('Exterior & Urban', '🏙️'), 'Airplane': ('Exterior & Urban', '🏙️'),
    'Fireworks': ('Exterior & Urban', '🏙️'), 'Hand saw': ('Exterior & Urban', '🏙️'),
}

CATEGORY_COLORS = {
    'Animals':             '#4f46e5',
    'Natural Soundscapes': '#0d9488',
    'Human Sounds':        '#d97706',
    'Interior & Domestic': '#6366f1',
    'Exterior & Urban':    '#e11d48',
}

# ====================================
# CSS - ML PORTFOLIO STYLE + TEXT FIX
# ====================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---- Base & Typography ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #f8fafc !important; 
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ---- PAKSA TEKS GELAP (FIX WHITE TEXT ISSUE) ---- */
p, span, div, h1, h2, h3, h4, h5, h6, label, li {
    color: #1e293b !important;
}
.stRadio label, .stRadio p, .stFileUploader label, .stFileUploader p {
    color: #334155 !important;
    font-weight: 600 !important;
}
[data-testid="stAlert"] p { color: #1e293b !important; font-weight: 500 !important; }

/* ---- FIX WIDGET SELECTBOX & DROPDOWN MENU (SUPER FORCE) ---- */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
}
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] ul,
ul[role="listbox"],
ul[data-testid="stVirtualDropdown"] {
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] li,
li[role="option"] {
    background-color: #ffffff !important;
    color: #1e293b !important;
    font-weight: 500 !important;
}
div[data-baseweb="popover"] li:hover,
li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #f1f5f9 !important;
    color: #4f46e5 !important; 
    font-weight: 700 !important;
}

/* ---- KUNCI SIDEBAR (STATIS) ---- */
/* 1. Hilangkan tombol panah/silang penutup agar TIDAK BISA diminimize manual */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* 2. Paksa tombol pembuka muncul jika tertutup otomatis (misal saat dibuka di HP) */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    z-index: 999999 !important;
}

/* 3. Kunci lebar sidebar dan berikan warna latar */
[data-testid="stSidebar"] {
    min-width: 270px !important;
    max-width: 270px !important;
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* 4. Hilangkan garis yang bisa digeser-geser oleh kursor */
[data-testid="stSidebarResizer"] { 
    display: none !important; 
}

/* ---- ML Portfolio Components ---- */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
    margin-top: -1.5rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #64748b !important;
    margin-bottom: 2rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.metric-value { font-size: 1.8rem; font-weight: 800; color: #4f46e5 !important; margin-bottom: 0.2rem; }
.metric-label { font-size: 0.85rem; font-weight: 600; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.5px; }
.pipeline-box {
    background: #ffffff;
    border-left: 4px solid #4f46e5;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    margin-bottom: 2rem;
}
.pipeline-step { font-weight: 600; color: #334155 !important; }

/* ---- Styling Footer Sidebar ---- */
.author-footer { text-align: center; margin-top: 3rem; font-size: 0.8rem; border-top: 1px dashed #cbd5e1; padding-top: 1.5rem; }
.author-name { color: #0f172a !important; font-weight: 700; font-size: 0.95rem; margin-top: 0.2rem; margin-bottom: 0.5rem; }
.tagline { font-size: 0.75rem; margin-bottom: 15px; color: #64748b !important; font-style: italic; }

header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================================
# TRANSFORMS & MODEL
# ====================================

mel_transform    = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0, normalized=True)
to_db            = T.AmplitudeToDB(stype='power', top_db=TOP_DB)
compute_delta    = T.ComputeDeltas(win_length=9)
imagenet_norm    = tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_transform = tv_transforms.Resize((224, 224))

@st.cache_resource
def load_model():
    model_path = "best_model.pth"
    
    # Trik: Download otomatis jika file model belum ada di server Streamlit Cloud
    if not os.path.exists(model_path):
        file_id = "1ztmOOnNKMM7ZobpexIi5zsNP5jGHwP0b" # <--- MASUKKAN ID FILE DI SINI
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    ckpt = torch.load(model_path, map_location="cpu")
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

model = load_model()

def preprocess_audio(path):
    audio_np = librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    mel = mel_transform(waveform)
    mel = to_db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    d1  = compute_delta(mel)
    d2  = compute_delta(d1)
    x   = torch.cat([mel, d1, d2], dim=0)
    x   = resize_transform(x)
    x   = imagenet_norm(x)
    return x.unsqueeze(0)

def predict(path):
    x = preprocess_audio(path)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred_idx   = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    top_probs, top_idx = torch.topk(probs, 5)
    return CLASS_NAMES[pred_idx], confidence, top_probs, top_idx

def plot_waveform(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    fig, ax = plt.subplots(figsize=(8, 1.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.plot(np.linspace(0, len(y)/sr, num=len(y)), y, color="#4f46e5", linewidth=0.8)
    ax.axis('off')
    return fig

# ====================================
# SIDEBAR (STATIS)
# ====================================

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>🎙️</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #0f172a; margin-top: 0; font-weight: 800; margin-bottom: 2rem;'>Environmental Audio Analysis</h3>", unsafe_allow_html=True)
    
    app_mode = st.radio("Navigation", ["⚡ Overview", "📂 Inference (File)"], label_visibility="collapsed")
    
    st.markdown("""
    <div class="author-footer">
        <span style="color: #64748b;">Designed by:</span><br>
        <div class="author-name">Daniel Lincoln Ginting</div>
    </div>
    """, unsafe_allow_html=True)

# ====================================
# MAIN UI ROUTING
# ====================================

col_left, col_main, col_right = st.columns([0.5, 8, 0.5])

with col_main:
    
    # --- HALAMAN BERANDA ---
    if app_mode == "⚡ Overview":
        st.markdown("<div class='hero-title'>Environmental Audio Analysis</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle'>Menerapkan Deep Learning untuk mengubah gelombang suara menjadi klasifikasi data terstruktur.</div>", unsafe_allow_html=True)
        
        # Metric Cards
        st.markdown("<div style='display: flex; gap: 1rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'><div class='metric-value'>84.00%</div><div class='metric-label'>Test Accuracy</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><div class='metric-value'>50</div><div class='metric-label'>Audio Classes</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'><div class='metric-value'>ResNet50</div><div class='metric-label'>CNN Architecture</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Methodology Pipeline
        st.markdown("<h4 style='font-weight: 700; margin-bottom: 1rem;'>⚙️ Data Processing Pipeline</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class='pipeline-box'>
            <p style='margin-top: 0; color: #475569;'>Sistem ini mengonversi masalah pemrosesan audio menjadi masalah <i>Computer Vision</i> menggunakan pendekatan Transfer Learning. Alur kerjanya meliputi:</p>
            <ol style='color: #334155; padding-left: 1.2rem; margin-bottom: 0;'>
                <li><span class='pipeline-step'>Resampling & Mono Conversion:</span> Penyeragaman frekuensi audio pada 44.1 kHz.</li>
                <li><span class='pipeline-step'>Feature Extraction:</span> Ekstraksi representasi spektral (Mel-Spectrogram).</li>
                <li><span class='pipeline-step'>Temporal Dynamics:</span> Pembentukan fitur <i>Delta</i> dan <i>Delta-Delta</i> untuk menangkap perubahan sinyal waktu.</li>
                <li><span class='pipeline-step'>3-Channel Input:</span> Penggabungan ketiga fitur menjadi representasi mirip citra RGB untuk diproses oleh <b>ResNet50</b>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # Dataset Info
        st.markdown("<h4 style='font-weight: 700; margin-bottom: 1rem;'>📊 ESC-50 Dataset Dimensions</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #475569;'>Model dilatih menggunakan dataset ESC-50 yang terdiri dari 2000 sampel rekaman lapangan, terdistribusi ke dalam 5 dimensi utama:</p>", unsafe_allow_html=True)
        
        tags = [
            ("Animals", "#4f46e5"), ("Natural Soundscapes", "#0d9488"), 
            ("Human Sounds", "#d97706"), ("Interior & Domestic", "#6366f1"), 
            ("Exterior & Urban", "#e11d48")
        ]
        
        tag_html = "".join([f"<span style='display: inline-block; background-color: {color}15; color: {color}; padding: 0.4rem 1rem; border-radius: 6px; font-weight: 600; font-size: 0.85rem; margin-right: 0.5rem; margin-bottom: 0.5rem; border: 1px solid {color}30;'>{name}</span>" for name, color in tags])
        st.markdown(f"<div>{tag_html}</div>", unsafe_allow_html=True)

    # --- HALAMAN INFERENCE (UPLOAD) ---
    else:
        st.markdown("<div class='hero-title'>File Inference</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-subtitle' style='margin-bottom: 1rem;'>Unggah file audio untuk mengevaluasi performa model.</div>", unsafe_allow_html=True)
        
        predict_btn = False
        temp_path = None

        input_type = st.radio("Pilih Sumber Data:", ["Gunakan Contoh Audio (Dataset ESC-50)", "Upload File Lokal (.wav)"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if input_type == "Upload File Lokal (.wav)":
            uploaded_file = st.file_uploader("Drop audio file here", type=["wav"])
            if uploaded_file is not None:
                st.audio(uploaded_file)
                if st.button("🚀 Execute Neural Network", use_container_width=True, type="primary"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                        predict_btn = True

        elif input_type == "Gunakan Contoh Audio (Dataset ESC-50)":
            example_dir = "examples"
            if os.path.exists(example_dir) and os.path.isdir(example_dir):
                example_files = [f for f in os.listdir(example_dir) if f.endswith('.wav')]
                if len(example_files) > 0:
                    selected_filename = st.selectbox("Pilih Sampel Uji:", example_files)
                    selected_example_path = os.path.join(example_dir, selected_filename)
                    st.audio(selected_example_path)
                    
                    if st.button("🚀 Execute Neural Network", use_container_width=True, type="primary"):
                        with open(selected_example_path, "rb") as f:
                            file_bytes = f.read()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(file_bytes)
                            temp_path = tmp.name
                            predict_btn = True
                else:
                    st.warning("⚠️ Folder 'examples' kosong.")
            else:
                st.warning("⚠️ Folder 'examples' tidak ditemukan di direktori proyek.")

        # --- HASIL PREDIKSI ---
        if temp_path and predict_btn:
            st.markdown("<hr style='border:1px solid #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
            with st.spinner("Processing 3-channel feature extraction..."):
                try:
                    label, confidence, top_probs, top_idx = predict(temp_path)
                    cat, cat_emoji = CATEGORY_MAP.get(label, ('Unknown', '❓'))
                    cat_color = CATEGORY_COLORS.get(cat, '#4f46e5')

                    res_col1, res_col2 = st.columns([1, 1.2])
                    
                    with res_col1:
                        st.markdown("<h4 style='font-weight: 700;'>Classification Result</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{cat_emoji}</div>
                            <div style="font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Predicted Class</div>
                            <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a !important; margin-bottom: 0.5rem;">{label.replace('_', ' ').title()}</div>
                            <div style="display: inline-block; background-color: {cat_color}15; color: {cat_color}; border: 1px solid {cat_color}40; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">{cat}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<h5 style='font-weight: 700; color: #334155; margin-top: 1.5rem;'>Temporal Waveform</h5>", unsafe_allow_html=True)
                        st.pyplot(plot_waveform(temp_path))

                    with res_col2:
                        st.markdown("<h4 style='font-weight: 700;'>Probability Distribution</h4>", unsafe_allow_html=True)
                        st.markdown("<div style='background: #ffffff; border: 1px solid #e2e8f0; padding: 1.5rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>", unsafe_allow_html=True)
                        
                        for i, (p, idx) in enumerate(zip(top_probs, top_idx)):
                            name    = CLASS_NAMES[idx.item()]
                            pct     = p.item()
                            c, _    = CATEGORY_MAP.get(name, ('Unknown', ''))
                            c_color = CATEGORY_COLORS.get(c, '#4f46e5')

                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                    <span style="font-weight: 600; color: #1e293b !important;">{name.replace('_', ' ').title()}</span>
                                    <span style="font-weight: 700; color: #4f46e5 !important;">{pct:.1%}</span>
                                </div>
                                <div style="background: #f1f5f9; border-radius: 999px; height: 8px; width: 100%;">
                                    <div style="background: {c_color}; height: 100%; border-radius: 999px; width: {pct*100:.1f}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan komputasi: {e}")
                finally:
                    os.unlink(temp_path)
