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

    # === Feature Engineering (Simplified) ===
    df['ratio_delay_durasi'] = df['delay_perpanjangan_kontrak'] / df['durasi_kontrak']
    df['nilai_per_hari'] = df['nilai_kontrak'] / df['durasi_kontrak']
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # ====== Model 1: Risk Level (Rule-based) ======
    def classify_risk(row):
        if row['nilai_kontrak'] > 5e9 or row['durasi_kontrak'] > 300 or row['delay_perpanjangan_kontrak'] > 30:
            return "Tinggi"
        elif row['nilai_kontrak'] > 2e9 or row['durasi_kontrak'] > 150 or row['delay_perpanjangan_kontrak'] > 0:
            return "Sedang"
        else:
            return "Rendah"
    df['Risk_Level'] = df.apply(classify_risk, axis=1)

    # ====== Model 2: Priority Level (ML) ======
    try:
        model_priority = joblib.load("model_priority_rf.pkl")
        fitur_model2 = ['nilai_kontrak', 'durasi_kontrak', 'delay_perpanjangan_kontrak']
        df['Prioritas'] = model_priority.predict(df[fitur_model2])
    except:
        df['Prioritas'] = "Model belum dimuat"

    # ====== Predicted Duration (Model 3 - Eksperimen) ======
    with st.expander("🧪 Lihat Hasil Eksperimen Model 3 (Prediksi Durasi Kontrak)"):
        try:
            model_duration = joblib.load("model_durasi_xgb.pkl")
            fitur_model3 = ['delay_perpanjangan_kontrak', 'nilai_kontrak', 'ratio_delay_durasi', 'nilai_per_hari']
            df['Predicted_Duration'] = model_duration.predict(df[fitur_model3])
            st.write("📉 Visualisasi Prediksi vs Aktual:")
            st.line_chart(df[['durasi_kontrak', 'Predicted_Duration']])
            st.caption("⚠️ Catatan: Model ini hanya untuk keperluan evaluatif, bukan untuk digunakan dalam sistem produksi.")
        except:
            st.warning("Model 3 belum tersedia. Pastikan file model_durasi_xgb.pkl tersedia di folder.")

    # ====== Visualisasi Dasbor Utama ======
    st.subheader("📊 Tabel Monitoring Kontrak")
    st.dataframe(df[['nama_vendor', 'jenis_pengadaan', 'nilai_kontrak', 'durasi_kontrak',
                     'delay_perpanjangan_kontrak', 'Risk_Level', 'Priority_Level']], use_container_width=True)

    st.subheader("🔔 Notifikasi Otomatis")
    df_alert = df[(df['Risk_Level'] == 'Tinggi') | (df['Priority_Level'] == 'Tinggi')]
    st.write("Kontrak Risiko atau Prioritas Tinggi:")
    st.dataframe(df_alert, use_container_width=True)

else:
    st.info("Silakan unggah file kontrak berformat CSV untuk memulai.")
