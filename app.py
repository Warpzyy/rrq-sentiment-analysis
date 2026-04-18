import sys
import types

try:
    import keras.src.legacy as legacy
except ImportError:
    mock_legacy = types.ModuleType('legacy')
    sys.modules['keras.src.legacy'] = mock_legacy
    sys.modules['keras.src.legacy.saving'] = types.ModuleType('saving')

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Analisis Sentimen RRQ", page_icon="🎮")

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_assets():
    try:
        # 1. Load Tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # 2. Bangun Rangka Model (Harus persis dengan saat training)
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=32),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # 3. Masukkan bobotnya
        model.load_weights('model_weights.weights.h5')
            
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file: {e}")
        return None, None

# --- FUNGSI PREDIKSI (HYBRID: Kamus + AI) ---
def prediksi_sentimen(teks):
    if model is None or tokenizer is None:
        return "Error", "⚠️"

    teks_clean = teks.lower()
    
    # Logika Kamus (Jurus Hybrid agar akurasi mantap)
    hujatan = ['lose', 'streak', 'cupu', 'jelek', 'kalah', 'beban', 'out', 'bubarkan', 'turu', 'cacat', 'aneh', 'ganti']
    pujian = ['menang', 'gacor', 'mantap', 'gg', 'king', 'semangat', 'bangkit', 'api', 'viva', 'kingdom', 'nice']
    
    # Cek Kamus dulu
    if any(word in teks_clean for word in hujatan):
        return 'Negatif', "🔴"
    elif any(word in teks_clean for word in pujian):
        return 'Positif', "🟢"
    
    # Jika tidak ada di kamus, baru tanya AI LSTM
    seq = tokenizer.texts_to_sequences([teks_clean])
    padded = pad_sequences(seq, maxlen=50, padding='post')
    pred = model.predict(padded, verbose=0)
    
    # Label: Negatif(0), Netral(1), Positif(2)
    labels = ['Negatif', 'Netral', 'Positif']
    hasil = labels[np.argmax(pred)]
    
    emoji = "🟢" if hasil == 'Positif' else "🔴" if hasil == 'Negatif' else "⚪"
    return hasil, emoji

# --- TAMPILAN DASHBOARD ---
st.title("🎮 RRQ Sentiment Analytics")
st.markdown("### Kingdom Voice: AI Sentiment Detection")
st.write("Menganalisis perasaan fans berdasarkan komentar menggunakan Deep Learning LSTM.")

st.divider()

input_user = st.text_area("Masukkan Komentar Fans:", placeholder="Contoh: RRQ Semangat, bangkit di match depan!")

if st.button("Analisis Sekarang"):
    if input_user:
        with st.spinner('AI sedang berpikir...'):
            hasil, emoji = prediksi_sentimen(input_user)
        
        st.subheader("Hasil Analisis:")
        if hasil == 'Positif':
            st.success(f"### {emoji} {hasil}")
        elif hasil == 'Negatif':
            st.error(f"### {emoji} {hasil}")
        else:
            st.info(f"### {emoji} {hasil}")
            
        st.divider()
        st.caption("Model: LSTM (Long Short-Term Memory) | Status: Production")
    else:
        st.warning("Ketik komentarnya dulu dong, hhe.")