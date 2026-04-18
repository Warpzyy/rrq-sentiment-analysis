import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Analisis Sentimen RRQ", page_icon="🎮")

# --- LOAD MODEL & TOKENIZER ---
# Kita asumsikan model sudah disimpan dengan nama ini
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_sentimen_rrq_final.keras')

try:
    model = load_my_model()
    # Load tokenizer yang kita simpan saat training tadi
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error: Pastikan file model dan tokenizer.pickle ada di folder! {e}")

# --- FUNGSI PREDIKSI (HYBRID) ---
def prediksi_sentimen(teks):
    teks_clean = teks.lower()
    
    # Logika Manual (Biar makin akurat)
    hujatan = ['lose', 'streak', 'cupu', 'jelek', 'kalah', 'beban', 'out', 'bubarkan', 'turu']
    pujian = ['menang', 'gacor', 'mantap', 'gg', 'king', 'semangat', 'bangkit', 'api']
    
    if any(word in teks_clean for word in hujatan):
        return 'Negatif', "🔴"
    elif any(word in teks_clean for word in pujian):
        return 'Positif', "🟢"
    
    # Kalau tidak ada di kamus, pakai AI LSTM
    seq = tokenizer.texts_to_sequences([teks_clean])
    padded = pad_sequences(seq, maxlen=50, padding='post')
    pred = model.predict(padded, verbose=0)
    
    labels = ['Negatif', 'Netral', 'Positif']
    hasil = labels[np.argmax(pred)]
    
    if hasil == 'Positif': emoji = "🟢"
    elif hasil == 'Negatif': emoji = "🔴"
    else: emoji = "⚪"
    
    return hasil, emoji

# --- TAMPILAN WEB ---
st.title("🎮 RRQ Sentiment Analytics")
st.write("Aplikasi AI untuk mendeteksi perasaan fans RRQ (Kingdom) berdasarkan komentar.")

st.divider()

input_user = st.text_area("Masukkan Komentar Fans di sini:", placeholder="Contoh: RRQ Gacor banget hari ini!")

if st.button("Analisis Sentimen"):
    if input_user:
        hasil, emoji = prediksi_sentimen(input_user)
        
        st.subheader("Hasil Analisis:")
        if hasil == 'Positif':
            st.success(f"{emoji} Sentimen Terdeteksi: **{hasil}**")
        elif hasil == 'Negatif':
            st.error(f"{emoji} Sentimen Terdeteksi: **{hasil}**")
        else:
            st.info(f"{emoji} Sentimen Terdeteksi: **{hasil}**")
            
        st.write("---")
        st.caption("Analisis ini dilakukan menggunakan model Deep Learning LSTM & Kamus Sentimen.")
    else:
        st.warning("Silakan ketik komentar dulu ya!")