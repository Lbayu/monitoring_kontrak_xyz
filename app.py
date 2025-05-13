import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb  # <== Tambah ini

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

    # ====== Load Encoders & Transform ======
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
        
        # Konversi Prioritas angka ke label
        reverse_map = {v: k for k, v in zip(le_priority.classes_, le_priority.transform(le_priority.classes_))}
        df['Prioritas_Label'] = df['Prioritas'].map(reverse_map)

        st.success("✅ Model Priority berhasil diprediksi.")
    except Exception as e:
        st.error(f"❌ Gagal memuat atau menjalankan model Priority: {e}")
        df['Prioritas_Label'] = "Model belum dimuat"

    # ====== Model 3: Prediksi Durasi Kontrak (Pakai .json) ======
    with st.expander("🧪 Lihat Hasil Eksperimen Model 3 (Prediksi Durasi Kontrak)"):
        try:
            model_duration = xgb.Booster()
            model_duration.load_model("model_durasi_xgb.json")
            feature_order = joblib.load("feature_order_model3.pkl")  # Urutan fitur waktu training

            df['Risk_encoded'] = le_risk.transform(df['Risk Level'])

            # Tangani jika Prioritas hasil model angka → label → encoded
            if df['Prioritas'].dtype in [np.int64, np.int32, np.float64]:
                reverse_map = {v: k for k, v in zip(le_priority.classes_, le_priority.transform(le_priority.classes_))}
                df['Prioritas_label'] = df['Prioritas'].map(reverse_map)
                df['Prioritas_encoded'] = le_priority.transform(df['Prioritas_label'])
            else:
                df['Prioritas_encoded'] = le_priority.transform(df['Prioritas'])

            df_model3_input = df[feature_order]
            dmatrix = xgb.DMatrix(df_model3_input)
            df['Predicted_Duration'] = model_duration.predict(dmatrix)

            st.write("📉 Visualisasi Prediksi vs Aktual:")
            st.line_chart(df[['durasi_kontrak', 'Predicted_Duration']])

        except Exception as e:
            st.warning(f"Model 3 belum tersedia atau gagal diproses: {e}")

    # ====== Tabel Monitoring ======
    st.subheader("📊 Tabel Monitoring Kontrak")
    st.dataframe(df[['nama_vendor', 'jenis_pengadaan', 'nilai_kontrak', 'durasi_kontrak',
                     'delay_perpanjangan_kontrak', 'Risk Level', 'Prioritas_label']], use_container_width=True)

    # ====== Notifikasi Risiko & Prioritas ======
    st.subheader("🔔 Notifikasi Otomatis")
    df_alert = df[(df['Risk Level'] == 'Tinggi') | (df['Prioritas_label'] == 'Tinggi')]
    st.write("Kontrak Risiko atau Prioritas Tinggi:")
    st.dataframe(df_alert, use_container_width=True)

    # ====== Ekspor ke CSV ======
    st.subheader("💾 Simpan ke CSV")
    filename = st.text_input("Nama file output", "hasil_prediksi_kontrak.csv")
    if st.button("📥 Simpan Data Hasil"):
        df.to_csv(filename, index=False)
        st.success(f"✅ File disimpan: {filename}")

else:
    st.info("Silakan unggah file kontrak berformat CSV untuk memulai.")
