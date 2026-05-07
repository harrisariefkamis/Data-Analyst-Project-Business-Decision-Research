# Riset Keputusan Bisnis Berbasis Data: Prediksi & Pencegahan Customer Churn

> **Proyek Data Analyst** — Membangun sistem prediksi churn pelanggan menggunakan pendekatan ilmiah berbasis data, mulai dari riset pasar, pemrosesan data, pemodelan machine learning, hingga rekomendasi strategis yang dapat langsung diterapkan oleh manajemen bisnis.

---

## Daftar Isi

- [Gambaran Umum Proyek](#gambaran-umum-proyek)
- [Fase I — Riset & Persiapan Data](#fase-i--riset--persiapan-data)
- [Fase II — Visualisasi & Pemahaman Bisnis](#fase-ii--visualisasi--pemahaman-bisnis)
- [Fase III — Pemodelan Machine Learning](#fase-iii--pemodelan-machine-learning)
- [Fase IV — Kesimpulan Strategis](#fase-iv--kesimpulan-strategis)
- [Ringkasan Insight Bisnis & Rekomendasi](#ringkasan-insight-bisnis--rekomendasi)
- [Strategi Retensi Berbasis Feature Importance](#strategi-retensi-berbasis-feature-importance)
- [Struktur Proyek](#struktur-proyek)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Penyimpanan & Pemuatan Model](#penyimpanan--pemuatan-model)

---

## Gambaran Umum Proyek

Customer churn adalah salah satu ancaman terbesar bagi keberlangsungan bisnis. Biaya mempertahankan pelanggan yang ada jauh lebih rendah dibandingkan biaya akuisisi pelanggan baru. Proyek ini bertujuan membangun sistem prediksi churn yang dapat mengidentifikasi pelanggan berisiko tinggi secara proaktif, sehingga tim bisnis dapat mengambil tindakan retensi yang tepat sasaran dan efisien.

| Metrik | Target |
|---|---|
| Recall | ≥ 0.95 (tidak ada churner yang terlewat) |
| Precision | ≥ 0.75 (meminimalkan false positive) |
| Algoritma Utama | Random Forest Classifier |
| Definisi Churn | Tidak ada transaksi selama ≥ 6 bulan |

---

## Fase I — Riset & Persiapan Data

### Bab 1 — Riset Pasar & Konteks Bisnis

Landasan strategis sebelum analisis dimulai.

- **Analisis Lanskap Pasar**
  Memetakan kondisi industri, tren churn rata-rata sektoral, dan benchmark kompetitor sebagai kerangka acuan evaluasi performa bisnis.

- **Identifikasi Permasalahan Bisnis**
  Mendefinisikan rumusan masalah secara kuantitatif: berapa besar dampak churn terhadap pendapatan, dan mengapa prediksi dini menjadi keputusan strategis yang kritis.

- **Rekomendasi & Kerangka Teknis**
  Menentukan pendekatan analitik yang tepat — pemilihan algoritma, metrik evaluasi bisnis, dan teknik visualisasi yang relevan dengan kebutuhan stakeholder.

---

### Bab 2 — Pemrosesan & Kualitas Data

Memastikan data bersih, valid, dan siap dianalisis.

- **Impor & Eksplorasi Data Awal**
  Memuat dataset, memahami struktur, tipe data, dan distribusi awal untuk mengenali pola serta anomali sejak dini.

- **Pembersihan Data (Data Cleansing)**
  Menangani outlier, duplikasi, dan inkonsistensi format data yang dapat mengganggu akurasi model prediksi.

- **Penanganan Nilai Kosong (Missing Value)**
  Menerapkan strategi imputasi yang tepat — mean, median, atau modus — berdasarkan karakteristik dan distribusi tiap variabel.

- **Pelabelan Churn & Validasi Data**
  Mendefinisikan label churn berdasarkan inaktivitas transaksi, memvalidasi konsistensi data, serta menghapus kolom yang tidak relevan secara bisnis.

---

## Fase II — Visualisasi & Pemahaman Bisnis

### Bab 3 — Analisis Visual & Wawasan Data

Mengubah angka menjadi cerita bisnis yang terukur dan mudah dipahami stakeholder.

| No. | Visualisasi | Tujuan Bisnis |
|---|---|---|
| 1 | Tren Akuisisi Pelanggan per Tahun | Mengukur laju pertumbuhan basis pelanggan |
| 2 | Volume Transaksi per Tahun | Mengidentifikasi tren aktivitas dan musiman |
| 3 | Rata-Rata Nilai Transaksi Berbayar per Tahun | Mengukur pergeseran daya beli pelanggan aktif |
| 4 | Proporsi Pelanggan Churn vs Aktif | Memahami skala permasalahan churn |
| 5 | Distribusi Kategori Transaksi | Menemukan pola konsumsi dominan |
| 6 | Rata-Rata Nilai Transaksi per Kategori | Dasar alokasi sumber daya pemasaran |

---

## Fase III — Pemodelan Machine Learning

### Bab 4 — Pembangunan & Evaluasi Model Prediktif

Dari data historis menjadi sistem prediksi bisnis yang dapat diandalkan.

#### 4.1 Penentuan Fitur & Target Variabel

Memilih variabel prediktor yang paling relevan secara bisnis dan menetapkan variabel target: status churn pelanggan (`1` = Churn, `0` = Aktif).

```python
features = [
    'last_transaction_days',   # Hari sejak transaksi terakhir
    'login_frequency',         # Frekuensi login bulanan
    'total_spend',             # Total pengeluaran historis
    'avg_monthly_spend',       # Rata-rata pengeluaran per bulan
    'support_ticket_count',    # Jumlah tiket support
    'transaction_category'     # Kategori transaksi dominan
]
target = 'is_churn'
```

#### 4.2 Pembagian Data Latih & Uji

Memisahkan dataset secara stratifikasi untuk memastikan representasi churn yang proporsional di kedua set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

#### 4.3 Pelatihan, Prediksi & Evaluasi Model Awal

Membangun baseline model, menghasilkan prediksi, dan mengukur performa awal sebagai titik acuan peningkatan.

#### 4.4 Visualisasi Confusion Matrix

Memetakan True Positive, False Positive, True Negative, dan False Negative untuk memahami pola kesalahan prediksi secara visual.

```
                  Prediksi Tidak Churn    Prediksi Churn
Aktual Tidak Churn       TN                    FP ⚠️
Aktual Churn             FN ❌                 TP ✅
```

#### 4.5 Metrik Evaluasi: Akurasi, Presisi & Recall

Menginterpretasikan trade-off antara presisi dan recall dalam konteks biaya bisnis.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
```

| Metrik | Nilai | Interpretasi Bisnis |
|---|---|---|
| Recall | 1.00 | Tidak ada churner yang terlewat |
| Precision | 0.67 | ~33% prediksi adalah false positive |
| F1-Score | 0.80 | Keseimbangan presisi dan recall |

#### 4.6 Pelatihan Lanjutan dengan Random Forest

Meningkatkan performa model menggunakan ensemble learning dengan hyperparameter tuning dan analisis feature importance.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
```

#### 4.7 Penyimpanan Model dengan Pickle

> **Key Insight:** Model Random Forest yang telah dilatih dapat disimpan (*serialized*) ke dalam file menggunakan modul `pickle`. Ini memungkinkan kita untuk memuat dan menggunakan kembali model tanpa perlu melatihnya ulang di masa mendatang — menghemat waktu komputasi secara signifikan dan memastikan konsistensi prediksi di lingkungan produksi.

**Mengapa ini penting secara bisnis?** Melatih Random Forest dengan ratusan estimator dan ribuan baris data membutuhkan waktu dan sumber daya komputasi yang tidak sedikit. Dengan menyimpan model yang sudah jadi, tim engineering dapat langsung mengintegrasikannya ke sistem CRM atau API prediksi real-time tanpa mengulangi proses pelatihan setiap kali dibutuhkan.

**Menyimpan model ke file:**

```python
import pickle

# Simpan model yang sudah dilatih ke file .pkl
with open('models/random_forest_churn.pkl', 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)

print("Model berhasil disimpan: models/random_forest_churn.pkl")
```

**Memuat kembali model untuk prediksi:**

```python
import pickle

# Muat model dari file tanpa perlu melatih ulang
with open('models/random_forest_churn.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Langsung gunakan untuk prediksi data baru
y_pred_new = loaded_model.predict(X_new)
y_prob_new = loaded_model.predict_proba(X_new)[:, 1]

print(f"Probabilitas churn pelanggan baru: {y_prob_new}")
```

**Praktik terbaik penyimpanan model:**

| Aspek | Rekomendasi |
|---|---|
| Format file | `.pkl` (pickle) atau `.joblib` (lebih efisien untuk array besar) |
| Penamaan file | Sertakan versi & tanggal: `rf_churn_v1_2024Q4.pkl` |
| Lokasi penyimpanan | Direktori `models/` yang terpisah dari notebook |
| Versioning | Simpan model lama sebelum menimpa dengan model baru |
| Validasi | Selalu verifikasi akurasi model yang dimuat sebelum deploy |

> **Alternatif dengan joblib** (direkomendasikan untuk model scikit-learn berukuran besar):
>
> ```python
> from joblib import dump, load
>
> # Simpan
> dump(grid_search.best_estimator_, 'models/random_forest_churn.joblib')
>
> # Muat kembali
> loaded_model = load('models/random_forest_churn.joblib')
> ```

---

## Fase IV — Kesimpulan Strategis

### Bab 5 — Temuan Utama & Rekomendasi Strategis

Mengubah hasil model menjadi keputusan bisnis yang konkret dan dapat dieksekusi.

#### Temuan Kunci (Key Insights)

1. **Recall sempurna (1.0)** — Model berhasil menangkap seluruh pelanggan yang benar-benar akan churn tanpa ada yang terlewat.
2. **False positive signifikan** — Sekitar 6.623 pelanggan diprediksi churn padahal tidak, sehingga berpotensi memboroskan anggaran kampanye retensi hingga ~33%.
3. **Presisi perlu ditingkatkan** — Target presisi ≥ 0.75 dapat dicapai melalui hyperparameter tuning dan penambahan fitur perilaku pelanggan.

#### Rekomendasi Strategis Berbasis Data

| Probabilitas Churn | Tier Risiko | Strategi Intervensi | Estimasi Biaya |
|---|---|---|---|
| ≥ 0.80 | Tinggi | Telepon personal, diskon eksklusif, account manager | Tinggi |
| 0.50 – 0.79 | Menengah | Email campaign personal, survei kepuasan | Sedang |
| 0.30 – 0.49 | Rendah | Newsletter otomatis, loyalty point reminder | Rendah |
| < 0.30 | Sangat Rendah | Tidak ada intervensi khusus | Minimal |

#### Peta Jalan Pengembangan ke Depan

- **Kuartal 1** — Integrasi model ke sistem CRM, mulai pengumpulan data feedback kampanye
- **Kuartal 2** — Evaluasi performa model di produksi, identifikasi concept drift
- **Kuartal 3** — Retrain model dengan data terbaru, eksplorasi fitur tambahan
- **Kuartal 4** — A/B testing strategi intervensi, optimasi ROI kampanye retensi

---

## Struktur Proyek

```
📁 customer-churn-prediction/
├── 📁 data/
│   ├── raw/                    # Data mentah original
│   └── processed/              # Data yang telah dibersihkan
├── 📁 notebooks/
│   ├── 01_market_research.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_data_visualization.ipynb
│   └── 04_modeling_evaluation.ipynb
├── 📁 src/
│   ├── data_processing.py
│   ├── visualization.py
│   └── modeling.py
├── 📁 models/
│   ├── random_forest_churn.pkl       # Model tersimpan (pickle)
│   └── random_forest_churn.joblib    # Model tersimpan (joblib)
│   └── figures/                # Hasil visualisasi
├── requirements.txt
└── README.md
```

---

## Teknologi yang Digunakan

| Kategori | Library / Tools |
|---|---|
| Bahasa Pemrograman | Python 3.10+ |
| Manipulasi Data | Pandas, NumPy |
| Visualisasi | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Algoritma Utama | Random Forest Classifier |
| Evaluasi Model | Classification Report, Confusion Matrix, ROC-AUC |
| Penyimpanan Model | Pickle, Joblib |
| Notebook | Jupyter Notebook / Google Colab |
| Version Control | Git & GitHub |

---

## Cara Menjalankan Proyek

```bash
# 1. Clone repository
git clone https://github.com/username/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan notebook secara berurutan
jupyter notebook notebooks/01_market_research.ipynb
```

---

## Ringkasan Insight Bisnis & Rekomendasi

### Perbandingan Performa Model

| Metrik | Regresi Logistik | Random Forest | Peningkatan |
|---|---|---|---|
| Akurasi | 0.6688 | 0.8134 | +14.46% |
| Presisi | 0.6688 | 0.8639 | +19.51% |
| Recall | 1.0000 | 0.8558 | -14.42% |

> **Kesimpulan:** Meskipun recall Random Forest sedikit lebih rendah, peningkatan presisi yang substansial menjadikannya pilihan unggul untuk operasional bisnis — upaya retensi menjadi lebih tepat sasaran dan tidak membuang sumber daya pada pelanggan yang sebenarnya tidak berisiko.

---

### Insight dari Analisis Data

**Akuisisi Pelanggan per Tahun**
Tren akuisisi pelanggan mengungkap puncak dan lembah pertumbuhan. Perusahaan perlu memahami faktor di balik pola ini untuk mengoptimalkan strategi pemasaran dan pengembangan produk secara tepat waktu.

**Rata-Rata Jumlah Transaksi per Tahun**
Analisis rata-rata transaksi tahunan mengidentifikasi periode dengan kinerja penjualan kuat maupun lemah, yang dapat dikaitkan dengan tren akuisisi atau peristiwa pasar tertentu.

**Proporsi Transaksi Churn vs. Tidak Churn**

| Segmen | Jumlah Pelanggan | Proporsi Pelanggan | Kontribusi Transaksi |
|---|---|---|---|
| Churn | 67.132 | 67% | 32.5% |
| Tidak Churn | 32.868 | 33% | 67.5% |

> Pelanggan yang tidak churn — meskipun jumlahnya lebih sedikit — menyumbang **67.5% dari total transaksi**. Ini menegaskan bahwa mempertahankan satu pelanggan aktif jauh lebih bernilai dibandingkan mencari pelanggan baru.

**Distribusi Kategori Produk**
- **Sepatu** — Kategori dengan total transaksi terbanyak
- **Jaket** — Kategori dengan rata-rata nilai transaksi tertinggi

Produk dengan rata-rata transaksi tinggi menjadi kandidat utama untuk kampanye retensi pelanggan bernilai tinggi.

---

### Rekomendasi Strategis

**1. Fokus pada Pelanggan Berisiko Tinggi**
Gunakan output model untuk memprioritaskan intervensi retensi:
- Model Regresi Logistik mengidentifikasi **13.377 pelanggan** berisiko churn
- Model Random Forest mengidentifikasi **11.448 pelanggan** berisiko churn (lebih presisi)

Program loyalitas, penawaran eksklusif, dan komunikasi personal harus difokuskan pada segmen ini terlebih dahulu.

**2. Analisis Akar Penyebab Churn (Root Cause)**
Gunakan fitur-fitur paling berpengaruh dalam model untuk menganalisis mengapa pelanggan churn — apakah ada pola perilaku atau demografi tertentu yang mendasarinya.

**3. Optimasi Strategi Berdasarkan Kategori Produk**
Identifikasi produk yang sering dibeli pelanggan berisiko churn. Tawarkan diskon, bundling, atau rekomendasi produk terkait untuk mencegah perpindahan.

**4. Bangun Program Loyalitas yang Kuat**
Perkuat program loyalitas untuk pelanggan aktif guna mendorong pembelian berulang dan membangun hubungan jangka panjang sebelum mereka masuk kategori berisiko.

**5. Pemantauan & Retrain Berkelanjutan**
Lakukan pemantauan rutin terhadap metrik churn dan performa model. Retrain secara berkala dengan data terbaru untuk menjaga akurasi dan relevansi prediksi.

**6. Eksplorasi Fitur Tambahan**
Pertimbangkan penambahan fitur: data demografi, riwayat penelusuran, interaksi layanan pelanggan, dan sentimen ulasan produk untuk meningkatkan akurasi model lebih lanjut.

---

## Strategi Retensi Berbasis Feature Importance

Analisis *feature importance* dari model Random Forest mengidentifikasi empat fitur utama yang paling berkontribusi terhadap prediksi churn:

```
Average_Transaction_Amount  ████████████████████  (tertinggi)
Year_Last_Transaction       ████████████████
Count_Transaction           ████████████
Year_First_Transaction      ████████
```

### 1. Average Transaction Amount — Rata-Rata Nilai Transaksi

Fitur dengan dampak paling signifikan. Pelanggan dengan rata-rata transaksi rendah lebih berisiko churn; pelanggan dengan nilai transaksi tinggi cenderung lebih loyal.

| Strategi | Implementasi |
|---|---|
| Program loyalitas berjenjang | Benefit lebih besar untuk pelanggan dengan transaksi lebih tinggi |
| Up-selling & cross-selling | Rekomendasi produk atau paket bundling untuk meningkatkan nilai belanja |
| Penawaran personalisasi | Diskon khusus berdasarkan riwayat pembelian individu |

### 2. Year Last Transaction — Tahun Transaksi Terakhir

Indikator terkuat kedua. Semakin lama jeda sejak transaksi terakhir, semakin tinggi probabilitas churn.

| Strategi | Implementasi |
|---|---|
| Kampanye reaktivasi | Email/SMS/notifikasi push untuk pelanggan yang tidak aktif 3–6 bulan |
| Survei pelanggan inaktif | Pahami alasan berhenti bertransaksi dan tawarkan solusi |
| Reminder produk | Ingatkan produk yang sering dibeli atau stok yang mungkin habis |

### 3. Count Transaction — Jumlah Transaksi

Mencerminkan frekuensi pembelian. Pelanggan dengan frekuensi rendah lebih rentan churn.

| Strategi | Implementasi |
|---|---|
| Program frekuensi pembelian | Reward setelah mencapai jumlah transaksi tertentu dalam satu periode |
| Komunikasi berkala | Konten relevan, tips produk, dan informasi promo untuk menjaga keterlibatan |
| Model langganan (subscription) | Pastikan pembelian berulang otomatis untuk produk yang memungkinkan |

### 4. Year First Transaction — Tahun Transaksi Pertama

Mengindikasikan loyalitas pelanggan berdasarkan lama hubungan dengan bisnis.

| Strategi | Implementasi |
|---|---|
| Program loyalitas berdasarkan usia pelanggan | Anniversary gift atau penawaran eksklusif untuk "anggota senior" |
| Segmentasi komunikasi | Pelanggan baru → edukasi produk; pelanggan lama → penawaran eksklusif & referral |
| Libatkan pelanggan lama | Ajak mereka dalam pengembangan produk/layanan baru sebagai mitra strategis |

---

## Penyimpanan & Pemuatan Model

Model yang telah dilatih disimpan di direktori `models/` menggunakan dua format:

- `random_forest_churn.pkl` — format pickle standar Python
- `random_forest_churn.joblib` — format joblib, lebih efisien untuk model scikit-learn berukuran besar

Pemuatan model untuk inferensi produksi cukup dilakukan dengan satu baris kode tanpa perlu mengulangi proses pelatihan:

```python
from joblib import load
model = load('models/random_forest_churn.joblib')
predictions = model.predict_proba(X_new)[:, 1]
```

---

> **Catatan:** Proyek ini dikembangkan sebagai bagian dari portofolio Data Analyst. Seluruh analisis bersifat reproduktif dan dapat dijalankan ulang dengan dataset yang sesuai.

---

*Dibuat dengan pendekatan data-driven untuk mendukung pengambilan keputusan bisnis yang lebih cerdas dan terukur.*
