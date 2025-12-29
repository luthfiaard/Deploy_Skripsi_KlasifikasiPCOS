import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# === Load model dan fitur ===
with open("finalmodel_klasifikasiPCOS.sav", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
selected_features = bundle["features"]

# File riwayat (CSV)
HISTORY_FILE = "riwayat_prediksi_pcos.csv"

def save_history_csv(data):
    df_new = pd.DataFrame([data])
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(HISTORY_FILE, index=False)

# === Judul dan instruksi ===
st.title("🧬 Prediksi PCOS dengan Random Forest")
st.write("Masukkan data berikut untuk melakukan prediksi:")
st.caption("⚠️ Jangan gunakan tanda koma (,) — gunakan tanda titik (.) untuk angka desimal. Bisa diisi lebih/kurang dari contoh range.")

# === Inisialisasi session state untuk riwayat ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Mapping deskripsi & contoh input ===
feature_info = {
    "Follicle No. (R)": {"desc": "Masukkan jumlah folikel di ovarium kanan", "range": "Contoh: 0 - 25"},
    "Follicle No. (L)": {"desc": "Masukkan jumlah folikel di ovarium kiri", "range": "Contoh: 0 - 25"},
    "Skin darkening (Y/N)": {"desc": "Apakah terdapat penggelapan kulit", "range": "Pilih: Tidak (0) / Ya (1)"},
    "Weight gain(Y/N)": {"desc": "Apakah terjadi peningkatan berat badan", "range": "Pilih: Tidak (0) / Ya (1)"},
    "hair growth(Y/N)": {"desc": "Apakah terjadi pertumbuhan rambut berlebih", "range": "Pilih: Tidak (0) / Ya (1)"},
    "Cycle(R/I)": {"desc": "Tipe siklus menstruasi", "range": "Pilih: Regular = Teratur (2) atau Irregular = Tidak Teratur (4)"},
    "AMH(ng/mL)": {"desc": "Masukkan nilai Anti-Müllerian Hormone", "range": "Contoh: 1 - 10"},
    "Cycle length(days)": {"desc": "Panjang siklus menstruasi (hari)", "range": "Contoh: 21 - 35"},
    "FSH(mIU/mL)": {"desc": "Masukkan nilai Follicle-Stimulating Hormone", "range": "Contoh: 3 - 15"},
    "LH(mIU/mL)": {"desc": "Masukkan nilai Luteinizing Hormone", "range": "Contoh: 2 - 20"}
}

# === Form input ===
st.markdown("### 🧾 Form Input Data")
user_input = {}

for feature in selected_features:
    info = feature_info.get(feature, {})
    st.markdown(f"**{feature}**  \nℹ️ {info.get('desc','')} ({info.get('range','')})")

    if feature in ["Skin darkening (Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)"]:
        pilihan = st.selectbox(
            feature, ["Pilih...", "Tidak (0)", "Ya (1)"],
            label_visibility="collapsed"
        )
        user_input[feature] = None if pilihan == "Pilih..." else (1.0 if "Ya" in pilihan else 0.0)

    elif feature == "Cycle(R/I)":
        pilihan = st.selectbox(
            feature, ["Pilih...", "Regular (0)", "Irregular (1)"],
            label_visibility="collapsed"
        )
        user_input[feature] = None if pilihan == "Pilih..." else (1.0 if "Irregular" in pilihan else 0.0)
    else:
        val = st.text_input(feature, "", label_visibility="collapsed")
        if val.strip() == "":
            user_input[feature] = None
        else:
            try:
                user_input[feature] = float(val.replace(",", "."))
            except ValueError:
                st.error(f"Input {feature} harus berupa angka!")
                user_input[feature] = None

# === Tombol aksi ===
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    pred_btn = st.button("🔍 Prediksi")
with col2:
    reset_btn = st.button("🔁 Reset Form Input")
with col3:
    history_btn = st.button("📊 Lihat Riwayat Prediksi (jika ada)")

# === Reset Form ===
if reset_btn:
    for feature in selected_features:
        if feature in st.session_state:
            del st.session_state[feature]
    st.rerun()

# === Jika tombol prediksi ditekan ===
if pred_btn:
    if any(v is None for v in user_input.values()):
        st.warning("⚠️ Harap isi semua data sebelum melakukan prediksi.")
    else:
        input_df = pd.DataFrame([user_input], columns=selected_features)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        st.subheader("📋 Data yang Diuji")
        for f, v in user_input.items():
            st.write(f"**{f}:** {v}")

        st.markdown("---")
        if prediction == 1:
            st.markdown(
                f"<h2 style='text-align:center;color:#FF4500;'>⚠️ Hasil: PCOS</h2>"
                f"<h3 style='text-align:center;'>Probabilitas: {probabilities[1]:.2%}</h3>",
                unsafe_allow_html=True
            )
            rekomendasi = "Sistem menyarankan untuk melakukan **konsultasi ke dokter spesialis kandungan** "
            "untuk pemeriksaan lebih lanjut."
        else:
            st.markdown(
                f"<h2 style='text-align:center;color:#2E8B57;'>💡 Hasil: Tidak PCOS</h2>"
                f"<h3 style='text-align:center;'>Probabilitas: {probabilities[0]:.2%}</h3>",
                unsafe_allow_html=True
            )
            rekomendasi = "Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin. "
            "Segera konsultasi ke dokter apabila muncul keluhan lain."

        st.info(f"🧾 **Rekomendasi Sistem:** {rekomendasi}")
        st.caption("⚠️ Sistem ini hanya berfungsi sebagai alat bantu prediksi, bukan diagnosis medis.")

        record = {
            "Prediksi": "PCOS" if prediction == 1 else "Tidak PCOS",
            "Probabilitas_PCOS": probabilities[1],
            "Probabilitas_Tidak_PCOS": probabilities[0],
            **user_input
        }

        st.session_state.history.append(record)
        save_history_csv(record)

        st.subheader("📊 Visualisasi Probabilitas")
        fig, ax = plt.subplots()
        ax.bar(["Tidak PCOS", "PCOS"], probabilities)
        ax.set_ylim(0, 1)
        for i, v in enumerate(probabilities):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center")
        st.pyplot(fig)

# === Tampilkan riwayat prediksi ===
if history_btn:
    st.subheader("📜 Riwayat Prediksi")
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df, use_container_width=True)

        with open(HISTORY_FILE, "rb") as f:
            st.download_button(
                "📥 Unduh Riwayat Prediksi (CSV)",
                data=f,
                file_name="riwayat_prediksi_pcos.csv",
                mime="text/csv"
            )
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")
