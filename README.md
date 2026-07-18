# Analisis Sentimen Bahasa Indonesia

Proyek ini melatih dan menjalankan model Analisis Sentimen Bahasa Indonesia berbasis TF-IDF dan Multinomial Naive Bayes. Semua data kebahasaan aktif berada di file resource, bukan di source Python, supaya preprocessing dapat diaudit dan dipakai konsisten oleh training, evaluasi, visualisasi, dan aplikasi Streamlit.

## Struktur Folder

- `pemodelan.py`: training, preprocessing bersama, penyimpanan model, dan metadata kompatibilitas.
- `evaluasi.py`: evaluasi model terlatih.
- `visualisasi.py`: pembuatan grafik dataset dan performa model.
- `app.py`: dashboard Streamlit.
- `project_config.py`: konfigurasi path dan loader konfigurasi.
- `setup_resources.py`: validasi integritas resource berdasarkan manifest.
- `clean_project.py`: pembersihan file pengembangan sebelum distribusi ZIP.
- `resources/`: konfigurasi dan resource bundled.
- `cache/`: resource aktif yang dapat diunduh bila hilang, misalnya `emoji_sentiment.csv`.
- `data1.csv` sampai `data5.csv`: dataset aktif.
- `tests/` dan `test_preprocessing.py`: test/audit otomatis.

## A. Instalasi

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Semua dependency di `requirements.txt` memiliki rentang versi. `nltk` tidak digunakan.

## B. Setup

```bash
python setup_resources.py
```

Mode normal hanya memvalidasi resource dan tidak memperbarui checksum manifest. Resource bundled wajib ikut dalam ZIP. Resource downloadable hanya diunduh bila file lokal belum tersedia. Setelah seluruh resource wajib tersedia secara lokal, aplikasi dapat berjalan tanpa internet.

Setup tidak selalu dapat membangun semua resource dari nol. Jika file bundled dihapus, kembalikan file dari distribusi asli atau backup yang valid.

Untuk memperbarui checksum setelah perubahan resource yang disengaja:

```bash
python setup_resources.py --update-checksums
```

Gunakan flag tersebut hanya setelah isi resource diverifikasi. Jika checksum mismatch terjadi pada mode normal, setup gagal dan manifest tidak diubah.

## C. Training

```bash
python pemodelan.py
```

Training menghasilkan `model_sentimen.pkl` dan `model_sentimen.metadata.json`. Metadata menyimpan checksum config/resource. Jika resource atau konfigurasi preprocessing berubah, model harus dilatih ulang.

## Evaluasi

```bash
python evaluasi.py
```

Evaluasi memakai preprocessing yang sama dengan training.

## Visualisasi

```bash
python visualisasi.py
```

Grafik disimpan ke `hasil_visualisasi/`.

## D. Menjalankan Aplikasi

```bash
python run.py
```

`run.py` memeriksa dependency, manifest, dan model. Script ini tidak menginstal dependency secara otomatis.

## E. Testing

```bash
pytest -v
python -m py_compile pemodelan.py evaluasi.py eda.py visualisasi.py app.py
python -m compileall .
python -m pytest -v
python cek_dataset.py
```

Jika executable `pytest` tidak tersedia di PATH, gunakan `python -m pytest -v`.

## F. Update Checksum Resmi

```bash
python setup_resources.py --update-checksums
```

Flag ini hanya digunakan ketika resource valid memang sengaja diubah. Mode update melewati perbandingan checksum lama, tetapi tetap menolak JSON rusak, CSV tanpa data, TXT kosong, resource hilang, schema salah, dan marker kerja.

## G. Membersihkan Proyek

```bash
python clean_project.py
```

Script ini menghapus folder/file pengembangan seperti `.venv/`, `.git/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.vscode/`, `.agents/`, `dataset_asli/`, `*.pyc`, `*.pyo`, `*.log`, `*.tmp`, dan `*.bak`. Script tidak menyentuh dataset aktif, model, metadata model, `resources/`, atau cache resource wajib.

## H. Distribusi ZIP

Yang boleh ikut ZIP:

- source Python aktif;
- `requirements.txt`;
- `README.md`;
- `.gitignore`;
- dataset `data*.csv`;
- `resources/`;
- `cache/emoji_sentiment.csv`;
- `model_sentimen.pkl`;
- `model_sentimen.metadata.json`;
- `test_preprocessing.py` dan folder `tests/`;
- `hasil_audit_konflik_label.csv` bila ingin menyertakan catatan pembersihan dataset.

Yang tidak boleh ikut ZIP:

- `.venv/`;
- `.git/`;
- `__pycache__/`;
- `.pytest_cache/`;
- `.mypy_cache/`;
- `.vscode/`;
- `.agents/`;
- `dataset_asli/`;
- `*.pyc`;
- `*.pyo`;
- `*.log`;
- `*.tmp`;
- `*.bak`.

## Troubleshooting

Checksum mismatch:

1. Jangan edit manifest secara manual untuk memaksa lolos.
2. Pulihkan resource dari distribusi yang valid, atau verifikasi perubahan resource.
3. Jika perubahan memang disengaja, jalankan `python setup_resources.py --update-checksums`.

Resource hilang:

1. Untuk bundled resource, salin kembali file dari ZIP/repository asli.
2. Untuk downloadable resource, jalankan `python setup_resources.py` dengan koneksi internet jika file lokal belum tersedia.

Model tidak kompatibel:

1. Jalankan `python setup_resources.py`.
2. Jalankan ulang `python pemodelan.py`.
3. Buka aplikasi kembali dengan `python run.py`.
