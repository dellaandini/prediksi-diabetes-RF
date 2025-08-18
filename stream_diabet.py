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
                        st.success("✅ Risiko Rendah")
                        st.write("""
                        **🩺🍎💪Saran Kesehatan:**
                        - Pertahankan pola hidup sehat
                        - Olahraga teratur minimal 30 menit/hari
                        - Konsumsi makanan bergizi dan seimbang
                        - Lakukan pemeriksaan rutin tahunan
                         """)
                        st.write("""
                        **🩺📅Pentingnya Pemeriksaan Berkala:**
                        \nMeskipun hasil saat ini risiko rendah, risiko bisa meningkat. Disarankan:
                        \n- Tes gula darah minimal setahun sekali jika memiliki faktor risiko
                        \n- Jaga berat badan ideal
                        \n- Monitor tekanan darah dan kolesterol
                        """)
                        st.write("""
                        **🤔🩸Apa Itu Pradiabetes?**
                        \nPradiabetes adalah kondisi di mana kadar gula darah lebih tinggi dari normal, tetapi belum cukup tinggi untuk dikatakan diabetes. 
                        Jika tidak ditangani, bisa berkembang menjadi diabetes tipe 2.                        
                        """)
                        # Disclaimer untuk Risiko Rendah
                        st.write("***Disclaimer:*** Hasil ini bukan pemeriksaan laboratorium resmi. Tetap jaga pola hidup sehat dan lakukan cek laboratorium bila mengalami tanda atau gejala tertentu.")
                    else:
                        st.error("⚠ Risiko Tinggi")
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
                         # Disclaimer untuk Risiko Tinggi
                        st.write("***Disclaimer:*** Hasil ini bukan pemeriksaan laboratorium resmi. Segera lakukan pemeriksaan medis lebih lanjut di fasilitas kesehatan dan konsultasikan dengan dokter.")

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
with col2:
    st.subheader("📖 Panduan Pengguna")
    with st.expander("Jumlah Kehamilan?"):
        st.write("""
        **Jumlah Kehamilan** adalah Jumlah kehamilan yang pernah dialami pasien. Semakin banyak kehamilan, dianggap berisiko karena terkait perubahan metabolik.
        - Diisi khusus untuk perempuan, berapa kali pernah hamil (termasuk keguguran/kelahiran mati). Untuk laki-laki isi 0.
        - Untuk laki-laki isi 0.
        """)
    with st.expander("Glukosa (mg/dL)"):
        st.write("""
        **Glukosa (mg/dL)** adalah kadar gula darah puasa Anda, yang biasanya diukur saat pemeriksaan laboratorium.
        - Kadar gula darah puasa Anda, yang biasanya diukur saat pemeriksaan laboratorium.
        - Hasil pemeriksaan gula darah puasa (Fasting Blood Sugar Test) di laboratorium/alat cek gula darah. Nilai normal: <100 mg/dL.
        """)
    with st.expander("Tekanan Darah (mmHg)"):
        st.write("""
        **Tekanan Darah (mmHg)** diukur dalam dua angka: tekanan sistolik (angka pertama) dan tekanan diastolik (angka kedua).
            Biasanya diperiksa menggunakan alat tensi meter.
        - Tekanan darah diastolik (mmHg). Tekanan darah tinggi dapat berkaitan dengan komplikasi diabetes.
        - Tekanan darah diukur dengan tensimeter/sphygmomanometer. Biasanya yang dipakai adalah tekanan darah diastolik (angka bawah). Normal: 80 mmHg.
        """)
    with st.expander("Ketebalan Kulit (mm)"):
        st.write("""
        **Ketebalan Kulit (mm)** mengukur ketebalan lipatan kulit pada bagian lengan atas. 
        - Ini juga biasa diukur di fasilitas kesehatan dengan menggunakan alat pengukur ketebalan kulit (caliper).  Satuan: mm. Jika tidak ada data, boleh diisi 0 sesuai dataset.
        - Ketebalan lipatan kulit triceps (mm). Mencerminkan kadar lemak tubuh.
            """)
    with st.expander("Insulin (μU/mL)"):
        st.write("""
        **Insulin (μU/mL)** adalah kadar insulin dalam darah yang biasanya diukur melalui pemeriksaan darah.
        - Kadar insulin serum 2 jam (µU/mL). Menunjukkan bagaimana tubuh memproduksi insulin. Nilai abnormal bisa menunjukkan diabetes atau prediabetes.
        - Hasil pemeriksaan laboratorium (tes OGTT/insulin darah). Jika tidak ada data, boleh diisi 0.
            """)
    with st.expander("BMI (Body Mass Index)"):
        st.write("""
        **BMI (Body Mass Index)** adalah Indeks massa tubuh, indikator obesitas yang merupakan faktor risiko kuat diabetes.
        - Hitung menggunakan rumus: BMI = Berat Badan (kg) ÷ [Tinggi Badan (m)]². 
        - Misal: 70 kg ÷ (1,60 m × 1,60 m) = 27,3 kg/m².
            """)
    with st.expander("Riwayat Diabetes Keluarga (Diabetes Pedigree Function / DPF)"):
        st.write("""
        **Riwayat Diabetes Keluarga (Diabetes Pedigree Function / DPF)** adalah Indikator seberapa kuat riwayat keluarga terhadap diabetes. Semakin tinggi nilai DPF, semakin tinggi risiko genetik.
Masukkan nilai DPF Anda berdasarkan riwayat keluarga dengan cara berikut:

Jika tidak ada riwayat keluarga dengan diabetes, isi 0.1.

Jika ada 1 anggota keluarga inti (ayah, ibu, atau saudara kandung) dengan diabetes, isi 0.4.

Jika ada 2 anggota keluarga inti dengan diabetes, isi 0.8.

Jika ada lebih dari 2 anggota keluarga inti (misalnya ayah, ibu, dan saudara kandung) dengan diabetes, isi 1.2.

👉 Rumus sederhana: Nilai DPF = 0.1 + (0.3 × jumlah anggota keluarga inti dengan diabetes)

Contoh:

Ibu saja → DPF = 0.4

Ayah dan kakak → DPF = 0.8

Ayah, ibu, dan adik → DPF = 1.2
            """)
        
    with st.expander("Usia (tahun)"):
        st.write("""
        **Usia (tahun)** adalah Usia pasien dalam tahun (21 tahun ke atas). Risiko diabetes meningkat seiring bertambahnya usia.
        - Diisi dengan umur saat ini (dalam tahun). 
            """)


# Footer
st.caption("📌 Aplikasi ini menggunakan model Random Forest terlatih berdasarkan *Pima Indian Diabetes Dataset*.")
