import sys
import types
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. JURUS ANTI-ERROR LEGACY ---
try:
    import keras.src.legacy as legacy
except ImportError:
    mock_legacy = types.ModuleType('legacy')
    sys.modules['keras.src.legacy'] = mock_legacy
    sys.modules['keras.src.legacy.saving'] = types.ModuleType('saving')

# --- 2. SETTING HALAMAN ---
st.set_page_config(page_title="Analisis Sentimen RRQ", page_icon="🎮")

# --- 3. FUNGSI LOAD (Didefinisikan Dulu) ---
@st.cache_resource
def load_assets():
    try:
        # 1. Load Tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # 2. Bangun Rangka Model & Tentukan Input Shape-nya
        model = tf.keras.Sequential([
            # Kita tambahkan input_shape di sini agar model langsung "terbentuk" (Built)
            tf.keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=50),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Paksa model untuk build agar bisa menerima bobot
        model.build(input_shape=(None, 50)) 
        
        # 3. Masukkan bobotnya
        model.load_weights('model_weights.weights.h5')
            
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file: {e}")
        return None, None

# --- 4. EKSEKUSI LOAD (Panggil Sekarang) ---
model, tokenizer = load_assets()

# --- 5. FUNGSI PREDIKSI ---
def prediksi_sentimen(teks):
    # Cek apakah model/tokenizer berhasil di-load
    if model is None or tokenizer is None:
        return "Error Load", "⚠️"

    teks_clean = teks.lower()
    
    # Hybrid: Cek Kamus Manual
    hujatan = ['lose', 'streak', 'cupu', 'jelek', 'kalah', 'beban', 'out', 'bubarkan', 'turu', 'cacat']
    pujian = ['menang', 'gacor', 'mantap', 'gg', 'king', 'semangat', 'bangkit', 'api', 'viva']
    
    if any(word in teks_clean for word in hujatan):
        return 'Negatif', "🔴"
    elif any(word in teks_clean for word in pujian):
        return 'Positif', "🟢"
    
    # AI LSTM
    seq = tokenizer.texts_to_sequences([teks_clean])
    padded = pad_sequences(seq, maxlen=50, padding='post')
    pred = model.predict(padded, verbose=0)
    
    labels = ['Negatif', 'Netral', 'Positif']
    hasil = labels[np.argmax(pred)]
    emoji = "🟢" if hasil == 'Positif' else "🔴" if hasil == 'Negatif' else "⚪"
    return hasil, emoji

# --- 6. TAMPILAN DASHBOARD ---
st.title("🎮 RRQ Sentiment Analytics")
st.markdown("### Kingdom Voice: AI Sentiment Detection")

st.divider()

input_user = st.text_area("Masukkan Komentar Fans:", placeholder="Contoh: RRQ Semangat!")

if st.button("Analisis Sekarang"):
    if input_user:
        hasil, emoji = prediksi_sentimen(input_user)
        
        st.subheader("Hasil Analisis:")
        if hasil == 'Positif':
            st.success(f"### {emoji} {hasil}")
        elif hasil == 'Negatif':
            st.error(f"### {emoji} {hasil}")
        else:
            st.info(f"### {emoji} {hasil}")
    else:
        st.warning("Ketik komentarnya dulu ya!")