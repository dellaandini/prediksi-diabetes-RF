import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Memuat model dan scaler
# Load model dan scaler
try:
    model = joblib.load('model_diabetes.pkl')
    scaler = joblib.load('scaler_diabetes.pkl')
except FileNotFoundError:
    st.error("❌ Model atau scaler tidak ditemukan. Pastikan file `model_diabetes.pkl` dan `scaler_diabetes.pkl` tersedia.")
    st.stop()


# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

# Fungsi untuk memuat logo universitas
@st.cache_data
def load_university_logo():
    logo_url = "https://drive.google.com/uc?export=download&id=1t8TVhW_S6gEQDCZmg9o18WYlSVlbnOH1"
    try:
        response = requests.get(logo_url)
        logo = Image.open(BytesIO(response.content))
        return logo
    except:
        st.warning("Gagal memuat logo universitas")
        return None

# Sidebar dengan logo dan identitas
logo = load_university_logo()
if logo:
    st.sidebar.image(logo, width=100)

st.sidebar.markdown("---")
st.sidebar.markdown("*Dibuat oleh:*")
st.sidebar.markdown("Della Andini")
st.sidebar.markdown("S1 Informatika Medis")
st.sidebar.markdown("---")

# Judul utama
st.title("🩺 Sistem Prediksi Diabetes")
st.markdown(
        "Aplikasi ini membantu memprediksi risiko diabetes berdasarkan data kesehatan dasar. "
        " Jaga kesehatan Anda dengan lebih baik! Inputkan data kesehatan Anda untuk mengetahui apakah Anda berisiko terkena diabetes."
    )

# Layout utama
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Masukkan Data Kesehatan")
    with st.form("diabetes_form"):
        col1a, col1b = st.columns(2)
        with col1a:
            pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, step=1)
        with col1b:
            glucose = st.number_input("Glukosa (mg/dL)", min_value=0, max_value=300, step=1)

        col2a, col2b = st.columns(2)
        with col2a:
            blood_pressure = st.number_input("Tekanan Darah (mmHg)", min_value=0, max_value=200, step=1)
        with col2b:
            skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0, max_value=100, step=1)

        col3a, col3b = st.columns(2)
        with col3a:
            insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=1000, step=1)
        with col3b:
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1, format="%.1f")

        col4a, col4b = st.columns(2)
        with col4a:
            diabetes_pedigree = st.number_input("Riwayat Diabetes Keluarga", min_value=0.0, step=0.001, format="%.3f")
        with col4b:
            age = st.number_input("Usia (tahun)", min_value=1, max_value=120, step=1)

        submitted = st.form_submit_button("Prediksi Diabetes")

        if submitted:
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age
            }

            # Validasi nilai nol (kecuali insulin dan ketebalan kulit)
            invalid = [k for k, v in input_data.items() if v == 0 and k not in ['Insulin', 'SkinThickness', 'Pregnancies']]
            if invalid:
                st.warning(f"Beberapa nilai penting tidak boleh nol: {', '.join(invalid)}. Silakan periksa kembali.")
            else:
                input_df = pd.DataFrame([input_data])

                try:
                    scaled_data = scaler.transform(input_df)
                    prediction = model.predict(scaled_data)
                    proba = model.predict_proba(scaled_data)[0]
                    confidence = proba[prediction[0]]

                    st.subheader("🔍 Hasil Prediksi")

                    if prediction[0] == 0:
                        st.success("✅ Risiko Rendah: Anda tidak terindikasi diabetes")
                        st.write("""
                        **🩺🍎💪Saran Kesehatan:**
                        - Pertahankan pola hidup sehat
                        - Olahraga teratur minimal 30 menit/hari
                        - Konsumsi makanan bergizi dan seimbang
                        - Lakukan pemeriksaan rutin tahunan
                         """)
                        st.write("""
                        **🩺📅Pentingnya Pemeriksaan Berkala:**
                        \nMeskipun hasil saat ini negatif, risiko bisa meningkat. Disarankan:
                        \n- Tes gula darah minimal setahun sekali jika memiliki faktor risiko
                        \n- Jaga berat badan ideal
                        \n- Monitor tekanan darah dan kolesterol
                        """)
                        st.write("""
                        **🤔🩸Apa Itu Pradiabetes?**
                        \nPradiabetes adalah kondisi di mana kadar gula darah lebih tinggi dari normal, tetapi belum cukup tinggi untuk dikatakan diabetes. 
                        Jika tidak ditangani, bisa berkembang menjadi diabetes tipe 2.                        
                        """)
                    else:
                        st.error("⚠ Risiko Tinggi: Anda berpotensi mengidap diabetes")
                        st.write("""
                        **🩺🍎💪Saran Kesehatan:**
                        - Jangan panik, ini baru prediksi awal.
                        - Segera konsultasikan ke dokter atau fasilitas kesehatan.
                        - Kurangi asupan gula dan karbohidrat olahan
                        - Tingkatkan aktivitas fisik secara konsisten
                        - Pantau kadar gula darah secara rutin
                        - Diskusikan dengan ahli gizi
                        """)
                        st.write("""
                        **🩸📉🍽️Manajemen Gula Darah:**
                        \n- Perhatikan pola makan: pilih karbohidrat kompleks, hindari gula tambahan.
                        \n- Makan dengan porsi kecil tapi sering (5–6 kali/hari).
                        \n- Rutin beraktivitas fisik
                        \n- Pantau kadar gula secara berkala (gunakan alat glukometer jika memungkinkan).
                        """)
                        st.write("""
                        **🩺⚠️🦶Komplikasi Diabetes:**
                        \nJika tidak dikendalikan, diabetes bisa menyebabkan:
                        \n- Gangguan penglihatan (retinopati)
                        \n- Penyakit ginjal
                        \n- Kerusakan saraf
                        \n- Luka sulit sembuh (terutama di kaki)
                        \n- Penyakit jantung dan stroke
                        """)
                        st.write("""
                        **🥗🚶‍♂️🧘Perubahan Gaya Hidup yang Disarankan:**
                        \n- Hindari makanan cepat saji dan minuman manis.
                        \n- Rutin berolahraga (minimal 150 menit/minggu).
                        \n- Kurangi stres (bisa lewat yoga, meditasi, hobi).
                        \n- Berhenti merokok dan kurangi alkohol.
                        """)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data: {e}")

with col2:
    st.subheader("📘 Tentang Diabetes")
    
    with st.expander("Apa itu diabetes?"):
        st.write("""
        Diabetes adalah penyakit gangguan metabolik yang ditandai dengan kadar gula darah tinggi.
        Ini terjadi karena pankreas tidak mampu memproduksi insulin secara cukup atau tubuh tidak efektif menggunakan insulin.
        Insulin adalah hormon penting yang mengatur metabolisme glukosa (gula) dalam darah.                 
        ✅ Jenis-jenis Diabetes:
        - Diabetes Tipe 1
        - Diabetes Tipe 2
        - Diabetes Gestasional
        - Diabetes Tipe Lain
        """)

    with st.expander("Faktor Resiko Diabets"):
        st.write("""
        - Berat badan berlebih (obesitas)
        - Gaya hidup tidak sehat
        - Kurang berolahraga
        - Faktor genetik
        - Riwayat penyakit tertentu
        """)


    with st.expander("Gejala Umum"):
        st.write("""
        Pada tahap awal, gejala diabetes sering tidak terlihat. Namun, seiring waktu, ada beberapa tanda umum yang perlu diwaspadai:
        - Sering buang air kecil
        - Berat badan turun drastis
        - Mudah lelah
        - Penglihatan kabur
        - Luka sulit sembuh
        """)

    with st.expander("Prediksi vs Diagnosis"):
        st.write("""
        Sistem ini memberikan prediksi berbasis data, bukan diagnosis medis. 
        Hanya dokter dan tes laboratorium yang bisa memastikan apakah seseorang benar-benar menderita diabetes.
        """)

# Footer
st.caption("📌 Aplikasi ini menggunakan model Random Forest terlatih berdasarkan *Pima Indian Diabetes Dataset*.")

