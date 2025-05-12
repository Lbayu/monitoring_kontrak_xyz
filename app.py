import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Dashboard Monitoring Kontrak IT PMA", layout="wide")
st.title("📊 Dashboard Monitoring Kontrak – IT PMA Bank XYZ")

# ===== Sidebar Upload =====
st.sidebar.header("📁 Upload File Kontrak")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview Data Kontrak")
    st.dataframe(df.head(), use_container_width=True)

    # === Feature Engineering ===
    df['ratio_delay_durasi'] = df['delay_perpanjangan_kontrak'] / df['durasi_kontrak']
    df['nilai_per_hari'] = df['nilai_kontrak'] / df['durasi_kontrak']
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # ====== Model 1: Risk Level (Rule-based) ======
    def classify_risk(row):
        if row['nilai_kontrak'] > 10e9 or row['durasi_kontrak'] > 300 or row['delay_perpanjangan_kontrak'] > 30:
            return "Tinggi"
        elif row['nilai_kontrak'] > 5e9 or row['durasi_kontrak'] > 180 or row['delay_perpanjangan_kontrak'] > 15:
            return "Sedang"
        else:
            return "Rendah"
    df['Risk Level'] = df.apply(classify_risk, axis=1)

    # ====== Load Encoders ======
    try:
        le_vendor = joblib.load("le_vendor.pkl")
        le_jenis = joblib.load("le_jenis.pkl")
        le_risk = joblib.load("le_risk_safe.pkl")
        le_priority = joblib.load("le_priority_safe.pkl")

        df['nama_vendor_encoded'] = le_vendor.transform(df['nama_vendor'])
        df['jenis_pengadaan_encoded'] = le_jenis.transform(df['jenis_pengadaan'])
    except Exception as e:
        st.error(f"❌ Gagal memuat encoder: {e}")

    # ====== Model 2: Priority Level (ML) ======
    try:
        model_priority = joblib.load("model_priority_rf.pkl")
        fitur_model2 = ['nilai_kontrak', 'durasi_kontrak', 'delay_perpanjangan_kontrak',
                        'jenis_pengadaan_encoded', 'nama_vendor_encoded']
        df['Prioritas'] = model_priority.predict(df[fitur_model2])
        st.success("✅ Model Priority berhasil diprediksi.")
    except Exception as e:
        st.error(f"❌ Gagal memuat atau menjalankan model Priority: {e}")
        df['Prioritas'] = "Model belum dimuat"

    # ====== Model 3: Predicted Duration (XGBoost) ======
    with st.expander("🧪 Lihat Hasil Eksperimen Model 3 (Prediksi Durasi Kontrak)"):
        try:
            model_duration = joblib.load("model_durasi_xgb.pkl")

            df['Risk_encoded'] = le_risk.transform(df['Risk Level'])
            df['Prioritas_encoded'] = le_priority.transform(df['Prioritas'])

            fitur_model3 = [
                'delay_perpanjangan_kontrak', 'nilai_kontrak',
                'ratio_delay_durasi', 'nilai_per_hari',
                'nama_vendor_encoded', 'jenis_pengadaan_encoded',
                'Risk_encoded', 'Prioritas_encoded'
            ]
            df['Predicted_Duration'] = model_duration.predict(df[fitur_model3])
            st.write("📉 Visualisasi Prediksi vs Aktual:")
            st.line_chart(df[['durasi_kontrak', 'Predicted_Duration']])
        except Exception as e:
            st.warning(f"Model 3 belum tersedia atau gagal diproses: {e}")

    # ====== Visualisasi Dasbor Utama ======
    st.subheader("📊 Tabel Monitoring Kontrak")
    st.dataframe(df[['nama_vendor', 'jenis_pengadaan', 'nilai_kontrak', 'durasi_kontrak',
                     'delay_perpanjangan_kontrak', 'Risk Level', 'Prioritas']], use_container_width=True)

    st.subheader("🔔 Notifikasi Otomatis")
    df_alert = df[(df['Risk Level'] == 'Tinggi') | (df['Prioritas'] == 'Tinggi')]
    st.write("Kontrak Risiko atau Prioritas Tinggi:")
    st.dataframe(df_alert, use_container_width=True)

else:
    st.info("Silakan unggah file kontrak berformat CSV untuk memulai.")
