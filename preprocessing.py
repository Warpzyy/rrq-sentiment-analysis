import pandas as pd

# Load data yang kamu kirim tadi
df = pd.read_csv('data_komentar_rrq_mpls17.csv')

# Lihat 5 data teratas
print(df.head())
import re
import string

def clean_text(text):
    # Ubah ke string untuk menghindari error jika ada data kosong
    text = str(text)
    # Case Folding: Ubah ke huruf kecil
    text = text.lower()
    # Menghapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Menghapus tanda baca (punctuation)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Menghapus whitespace berlebih
    text = text.strip()
    return text

# Kita asumsikan kolom komentar di CSV kamu bernama 'ytAttributedStringHost' 
# atau sesuaikan dengan nama kolom teks di hasil print(df.head()) tadi.
# Contoh jika kolomnya bernama 'ytAttributedStringHost':
df['text_clean'] = df.iloc[:, 5].apply(clean_text) # Mengambil kolom ke-6 (indeks 5) biasanya berisi komentar

print(df[['text_clean']].head())

# Cek jumlah total data
print(f"Total komentar: {len(df)}")

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi Sastrawi
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def preprocess_v2(text):
    # Stopword Removal
    text = stopword.remove(text)
    # Stemming
    text = stemmer.stem(text)
    return text

# Jalankan (Proses ini mungkin agak lama karena ada 7.000 data)
print("Sedang melakukan stemming... mohon tunggu.")
df['text_preprocessed'] = df['text_clean'].apply(preprocess_v2)

print(df[['text_preprocessed']].head())

# Tambahkan kolom platform
df['platform'] = 'YouTube'

# Buat kolom sentimen kosong untuk kamu isi manual nanti (Minimal 500-1000 data)
# Atau kita bisa buat simulasi label sederhana berdasarkan kata kunci (opsional)
def simple_labeler(text):
    pos_words = ['semangat', 'menang', 'bangkit', 'king', 'nice', 'gg']
    neg_words = ['kalah', 'lose', 'streak', 'bad', 'bubarkan', 'payah', 'noob']
    
    if any(word in text for word in neg_words):
        return 'Negatif'
    elif any(word in text for word in pos_words):
        return 'Positif'
    else:
        return 'Netral'

df['sentimen'] = df['text_preprocessed'].apply(simple_labeler)

# Simpan hasil sementara ke CSV baru agar tidak hilang
df.to_csv('data_rrq_preprocessed.csv', index=False)
print("Selesai! File data_rrq_preprocessed.csv telah dibuat.")