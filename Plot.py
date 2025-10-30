import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import cv2
import pandas as pd
import matplotlib.patches as patches
from function import read_img_as_interferogram


def save_uploaded_file(tmp_file, bytes_data):
    with open(tmp_file, "wb") as f:
        f.write(bytes_data)
    return tmp_file


def main():
    st.title("Interferogram 📷➡️📈")
    st.write("🔬• **Gambar** → ambil 1 baris lalu plot intensitas.")
    st.write("🎥• **Video** → ambil 1 piksel / 1 area dari tiap frame lalu plot intensitas vs waktu, plus deteksi peak.")

    mode = st.radio("Pilih mode:", ["Gambar (foto)", "Video (intensitas vs waktu)"])

    # =====================================================
    # MODE GAMBAR
    # =====================================================
    if mode == "Gambar (foto)":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Rotasi
            toggle_rotasi = st.checkbox("Rotasi gambar?")
            if toggle_rotasi:
                angle = st.slider("Sudut rotasi (derajat)", -180, 180, 0)
                image = image.rotate(angle, expand=True)
                st.image(image, caption=f"Setelah rotasi {angle}°", use_column_width=True)

            # Crop horizontal
            st.subheader("Crop gambar (horizontal)")
            values = st.slider(
                "Pilih batas kiri-kanan (px)",
                0.0,
                float(image.width - 1),
                (0.0, float(image.width - 1))
            )
            left, right = int(values[0]), int(values[1])
            if left >= right:
                st.error("Nilai kiri harus lebih kecil dari kanan.")
                return

            cropped_image = image.crop((left, 0, right, image.height))
            st.image(cropped_image, caption="Setelah crop", use_column_width=True)

            # pilih baris
            row = st.number_input(
                "Baris yang akan diplot (0 = paling atas)",
                min_value=0,
                max_value=int(cropped_image.height),
                value=int(cropped_image.height / 2)
            )

            # pakai fungsi kamu
            fig, df = read_img_as_interferogram(uploaded_file, int(row))
            st.pyplot(fig)

            # download plot
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="📥 Download plot (PNG)",
                data=buf,
                file_name="interferogram.png",
                mime="image/png"
            )

            # download csv
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download data intensitas (CSV)",
                data=csv,
                file_name="data_intensitas.csv",
                mime="text/csv"
            )

    # =====================================================
    # MODE VIDEO
    # =====================================================
    else:
        st.write("Mode ini memantau **1 titik / 1 area kecil** dari video lalu plot intensitasnya berdasarkan pixel.")
        uploaded_video = st.file_uploader("Upload video interferogram...", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_video is not None:
            st.video(uploaded_video)

            # simpan sementara
            tmp_path = "temp_video_input.mp4"
            save_uploaded_file(tmp_path, uploaded_video.getbuffer())

            # baca 1 frame dulu
            cap0 = cv2.VideoCapture(tmp_path)
            ok, frame0 = cap0.read()
            cap0.release()

            if not ok:
                st.error("❌ Tidak bisa membaca frame pertama dari video.")
                return

            frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            h0, w0, _ = frame0_rgb.shape
            st.write(f"Resolusi video: **{w0} x {h0}** px")

            st.subheader("Pilih titik / area yang mau dipantau")
            col1, col2 = st.columns(2)
            with col1:
                y_row = st.number_input("Baris (y)", min_value=0, max_value=h0 - 1, value=h0 // 2)
            with col2:
                x_col = st.number_input("Kolom (x)", min_value=0, max_value=w0 - 1, value=w0 // 2)

            use_roi = st.checkbox("Gunakan area kecil", value=True)
            roi_size = 5
            if use_roi:
                roi_size = st.slider("Ukuran area (ganjil)", 3, 151, 7, step=2)

            frame_step = st.slider("Ambil setiap n frame", 1, 10, 1)

            # =============== PREVIEW DENGAN TITIK + KOTAK ===============
            preview_frame = frame0_rgb.copy()
            fig_preview, ax_preview = plt.subplots(figsize=(6, 4))
            ax_preview.imshow(preview_frame)
            ax_preview.set_title("Frame pertama (buat referensi area)")
            ax_preview.set_axis_off()

            # titik pusat
            ax_preview.plot(
                x_col, y_row,
                "o",
                color="red",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="Titik pusat"
            )

            if use_roi:
                half = roi_size // 2
                x0 = x_col - half
                y0 = y_row - half

                # lapis bawah hitam
                ax_preview.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        roi_size,
                        roi_size,
                        linewidth=5,
                        edgecolor="black",
                        facecolor="none"
                    )
                )
                # lapis atas kuning transparan
                ax_preview.add_patch(
                    patches.Rectangle(
                        (x0, y0),
                        roi_size,
                        roi_size,
                        linewidth=2.5,
                        edgecolor="yellow",
                        facecolor=(1, 1, 0, 0.2)
                    )
                )

                # label kecil
                ax_preview.text(
                    x0,
                    max(0, y0 - 8),
                    f"{roi_size}x{roi_size}",
                    color="white",
                    fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none")
                )

            ax_preview.legend(loc="upper right", framealpha=0.6)
            st.pyplot(fig_preview)

            # st.write(f"Titik yang dipilih: (x, y) = ({int(x_col)}, {int(y_row)})")
            st.markdown(f"Titik yang dipilih: (x, y) = `({int(x_col)}, {int(y_row)})`")

            # =============== PROSES VIDEO ===============
            cap = cv2.VideoCapture(tmp_path)
            intensities = []
            frame_indices = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape

                yy = min(max(0, int(y_row)), h - 1)
                xx = min(max(0, int(x_col)), w - 1)

                if use_roi:
                    half = roi_size // 2
                    y1 = max(0, yy - half)
                    y2 = min(h, yy + half + 1)
                    x1 = max(0, xx - half)
                    x2 = min(w, xx + half + 1)
                    patch = gray[y1:y2, x1:x2].astype(np.float32)
                    val = patch.mean()
                else:
                    val = float(gray[yy, xx])

                intensities.append(val)
                frame_indices.append(frame_idx)
                frame_idx += 1

            cap.release()

            if len(intensities) == 0:
                st.error("❌ Tidak ada frame yang berhasil diproses.")
                return

            intensities = np.array(intensities, dtype=np.float32)

            # =============== NORMALISASI OPSIONAL ===============
            norm = st.checkbox("Normalisasi intensitas ke 0–1 ?", value=True)
            if norm:
                mn, mx = intensities.min(), intensities.max()
                if mx - mn > 1e-6:
                    intensities = (intensities - mn) / (mx - mn)

            # Deteksi peak
            st.subheader("Deteksi peak (laser ON) dengan threshold")
            threshold = st.slider("Threshold deteksi peak", 0.0, 1.0, 0.8, 0.01)
            binary_peak = (intensities >= threshold).astype(int)

            transitions = np.sum((binary_peak[1:] == 1) & (binary_peak[:-1] == 0))
            st.write(f"🔢 Jumlah peak (transisi 0→1): **{transitions}**")

            # Plot Intensitas vs Waktu
            st.subheader("Intensitas vs waktu (frame)")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(frame_indices, intensities, label="Intensitas")
            ax.axhline(threshold, color="red", linestyle="--", alpha=0.5, label=f"Threshold={threshold:.2f}")
            ax.set_xlabel("Frame / waktu")
            ax.set_ylabel("Intensitas")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # =============== PLOT BINER ===============
            st.subheader("Sinyal biner (1 = peak, 0 = non-peak)")
            fig_bin, ax_bin = plt.subplots(figsize=(8, 2.5))
            ax_bin.step(frame_indices, binary_peak, where="post")
            ax_bin.set_ylim(-0.2, 1.2)
            ax_bin.set_xlabel("Frame / waktu")
            ax_bin.set_ylabel("Peak")
            ax_bin.grid(True, alpha=0.3)
            st.pyplot(fig_bin)

            # download data
            df = pd.DataFrame({
                "frame": frame_indices,
                "intensity": intensities
            })
            st.download_button(
                "📥 Download data intensitas (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="intensitas_vs_waktu.csv",
                mime="text/csv"
            )

            # =============== DOWNLOAD CSV 2 (baru: + peak) ===============
            df_bin = pd.DataFrame({
                "frame": frame_indices,
                "intensity": intensities,
                "peak": binary_peak
            })
            st.download_button(
                "📥 Download data + peak biner (CSV)",
                data=df_bin.to_csv(index=False).encode("utf-8"),
                file_name="intensitas_vs_waktu_biner.csv",
                mime="text/csv"
            )

            # =============== DOWNLOAD PLOT ===============
            buf_plot = BytesIO()
            fig.savefig(buf_plot, format="png", bbox_inches="tight")
            buf_plot.seek(0)
            st.download_button(
                "📥 Download plot intensitas (PNG)",
                data=buf_plot,
                file_name="plot_intensitas.png",
                mime="image/png"
            )

            # hapus file sementara
            try:
                os.remove(tmp_path)
            except:
                pass


if __name__ == "__main__":
    main()
