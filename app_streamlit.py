import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from predict_tf import predict_text  # fungsi prediksi kamu

# ======== CONFIG ========
st.set_page_config(page_title="Emotion & Urgency Analyzer", layout="wide")

# ======== BACKGROUND IMAGE ========
def add_bg_from_local(image_path):
    with open(image_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    .main-card {{
        background: rgba(255, 255, 255, 0.85);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        backdrop-filter: blur(6px);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_from_local(r"C:\Users\ivt_0\Downloads\pexels-markusspiske-1089438.jpg")


# ======== HEADER ========


# ======== USER GUIDE ========
with st.expander("📘 Petunjuk Penggunaan", expanded=False):
    st.markdown("""
    1️⃣ **Prediksi Kalimat:** Ketik satu ulasan dan tekan tombol *Prediksi*  
    2️⃣ **Prediksi CSV:** Unggah file CSV yang memiliki kolom teks seperti `ulasan_bersih` atau `review`  
    3️⃣ Lihat hasil prediksi emosi dan urgensi disertai tingkat *confidence*  
    4️⃣ Hasil bisa diunduh sebagai CSV dan divisualisasikan dalam *pie chart*
    """)

# ======== TABS ========
tab1, tab2 = st.tabs(["Prediksi Kalimat", "Prediksi CSV"])

# --- Tab 1: Single Prediction
with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🧠 Prediksi Satu Kalimat")

    teks = st.text_area("Masukkan ulasan produk:", height=120,
                        placeholder="contoh: barang tidak sesuai deskripsi dan baterainya cepat habis")

    if st.button("Prediksi", type="primary"):
        if teks.strip():
            emosi, urg, conf_emosi, conf_urg = predict_text(teks)  # pastikan fungsi return 4 nilai
            col1, col2 = st.columns(2)
            col1.metric("Emosi", emosi)
            col2.metric("Urgensi", urg)

            # Confidence bars
            st.write("### Tingkat Confidence")
            st.progress(conf_emosi)
            st.caption(f"Confidence Emosi: {conf_emosi*100:.2f}%")
            st.progress(conf_urg)
            st.caption(f"Confidence Urgensi: {conf_urg*100:.2f}%")

            # Pie Chart
            st.write("### Visualisasi Distribusi Prediksi")
            fig, ax = plt.subplots()
            ax.pie([conf_emosi, conf_urg], labels=["Emosi", "Urgensi"],
                   autopct='%1.1f%%', startangle=90, colors=["#8fd3f4", "#84fab0"])
            st.pyplot(fig)
        else:
            st.warning("Silakan isi teks ulasannya terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: CSV Prediction
with tab2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📦 Prediksi Batch dari CSV")

    f = st.file_uploader("Upload file CSV:", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        text_col = next((c for c in ["ulasan_bersih", "text", "message", "review", "ulasan"] if c in df.columns), None)
        if text_col:
            results = []
            for s in df[text_col].astype(str).fillna(""):
                emosi, urg, conf_emosi, conf_urg = predict_text(s)
                results.append((s, emosi, urg, conf_emosi, conf_urg))
            out = pd.DataFrame(results, columns=[text_col, "emosi_pred", "urgensi_pred", "conf_emosi", "conf_urg"])

            st.success(f"Selesai memprediksi {len(out)} baris.")
            st.dataframe(out.head(20), use_container_width=True)

            st.download_button(
                "💾 Unduh Hasil CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

            # Pie Chart
            st.write("### Distribusi Emosi (Pie Chart)")
            fig2, ax2 = plt.subplots()
            out["emosi_pred"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, cmap="viridis")
            ax2.set_ylabel("")
            st.pyplot(fig2)
        else:
            st.error("Kolom teks tidak ditemukan. Harap sertakan kolom 'ulasan_bersih' atau 'review'.")
    st.markdown('</div>', unsafe_allow_html=True)
