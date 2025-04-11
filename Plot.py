import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO #untuk download plot
from function import read_img_as_interferogram

def image_to_bytes(image):
    # Konversi PIL Image ke bytes
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def main():
    st.title("Plot Interferogram 🔍")
    st.write("Unggah foto, crop, rotasi, dan lihat plot distribusi intensitas")

    # Unggah gambar
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Buka gambar dengan PIL
        image = Image.open(uploaded_file)

        # Rotasi gambar
        toggle_rotasi = st.checkbox("Tombol untuk Rotasi Gambar")
        if toggle_rotasi:
          st.subheader("Rotasi Gambar")
          rotation_angle = st.slider("Pilih sudut rotasi (derajat)", -180, 180, 0)
          rotated_image = image.rotate(rotation_angle, expand=True)
          st.write("rotasi - : ", int(rotation_angle))
          plt.imshow(rotated_image)
          plt.axis('equal')
          st.pyplot(plt)
        else :
          rotated_image = image

        # Crop gambar
        st.subheader("Crop Gambar")


        values = st.slider("pilih kiri kanan foto", 0.0, float(rotated_image.width-1), (0.0, float(rotated_image.width-1)))
        st.write("Values:", values)

        left = int(values[0])
        right = int(values[1])
        # Validasi input crop
        if left >= right:
            st.error("Nilai kiri harus lebih kecil dari kanan")
        else:
            # Lakukan crop
            cropped_image = rotated_image.crop((left, 0, right, rotated_image.height))
            
            plt.imshow(cropped_image)
            plt.axis('equal')
            st.pyplot(plt)

            # st.image(cropped_image, caption=f"Setelah crop: {cropped_image.width} x {cropped_image.height} piksel", use_column_width=True)

            # Tampilkan informasi koordinat mouse saat hover
            st.subheader("Visualisasi Intensitas ternormalisasi")
            st.write("Hasil ini akan memberikan hasil plot dari pola fringe berdasarkan nilai grayscale foto")

            # Konversi gambar ke array numpy untuk interaktivitas
            img_array = np.array(cropped_image)

            # st.write(f"Koordinat crop: X={left}-{right}px")
            # st.write(f"Dimensi gambar terakhir: Lebar (X) = {cropped_image.width}px")

            # row = st.slider("Pilih baris pixel", 0, int(cropped_image.height), int(cropped_image.height/2))
            row = st.number_input(
                  "Masukkan baris", value=int(cropped_image.height/2), placeholder="Baris ke ..."
              )
            st.write("baris ke - : ", int(row))
            if int(row)>int(cropped_image.height):
              st.error("Baris harus lebih kecil dari tinggi gambar")
            else :
              fig = read_img_as_interferogram(uploaded_file,row)
              st.pyplot(fig)
              # Tambahkan download button untuk gambar hasil edit
               # Simpan fig ke dalam objek BytesIO
              buf = BytesIO()
              fig.savefig(buf, format="png")
              buf.seek(0)

              # Tampilkan tombol download
              st.download_button(
                  label="Download Plot sebagai Gambar",
                  data=buf,
                  file_name="interferogram.png",
                  mime="image/png"
              )
if __name__ == "__main__":
    main()
