README - Dog vs Cat Image Classification

Deskripsi:
-----------
Aplikasi klasifikasi gambar Dog vs Cat menggunakan model TensorFlow (.h5).

Fitur:
------
1. Upload gambar tunggal
2. Prediksi kelas (Dog atau Cat)
3. Visualisasi probabilitas
4. Download hasil prediksi CSV (untuk batch)

Persyaratan Sistem:
------------------
- Python 3.10 atau lebih tinggi
- Pip

Dependencies:
-------------
streamlit==1.26.0
tensorflow==2.16.0
numpy==1.26.1
pandas==2.1.0
plotly==5.17.0
Pillow==10.1.0

Instalasi dependencies:
----------------------
pip install -r requirements.txt

File penting:
-------------
1. `app.py`                  -> script utama Streamlit
2. `Dog_vs_cat_model.h5`     -> model Dog vs Cat
3. `requirements.txt`        -> daftar library + versi

Cara Menjalankan (Local):
-------------------------
1. Pastikan dependencies sudah terinstall.
2. Simpan semua file di satu folder.
3. Jalankan:

streamlit run app.py

4. Browser akan otomatis membuka dashboard Streamlit.

Catatan:
--------
- Jangan ubah nama file model, path sudah diatur di `app.py`.
- Visualisasi muncul di sidebar dan halaman utama.

Support:
--------
Jika ada error terkait package, pastikan versi sesuai requirements.txt.

