# Home Credit Indonesia — Credit Risk Scoring

Project ini merupakan analisis dan pemodelan credit risk untuk membantu memprediksi probabilitas nasabah gagal bayar (`TARGET = 1`). Analisis tidak hanya berfokus pada performa model, tetapi juga mempertimbangkan dampaknya terhadap keputusan pemberian pinjaman, terutama keseimbangan antara mengurangi risiko gagal bayar dan tetap memberikan akses kepada nasabah yang mampu melunasi.

## Business Problem

Home Credit perlu menggunakan data yang tersedia untuk membantu proses penilaian risiko kredit. Tantangannya adalah mengurangi kerugian akibat gagal bayar tanpa terlalu banyak menolak nasabah yang sebenarnya mampu melunasi.

Selain keputusan approve/reject, hasil analisis juga digunakan sebagai dasar rekomendasi penyesuaian pemberian pinjaman berdasarkan tingkat risiko.

## Objective

1. Membangun model klasifikasi untuk memprediksi probabilitas nasabah gagal bayar.
2. Membandingkan beberapa model Machine Learning, termasuk Logistic Regression.
3. Mengidentifikasi karakteristik nasabah yang berkaitan dengan risiko gagal bayar.
4. Mengevaluasi dampak model terhadap approval rate, default rate, dan net outcome.

## Dataset

Dataset terdiri dari tabel utama `application_train` dan beberapa tabel yang berisi riwayat kredit nasabah:

* `application_train` — data utama pengajuan pinjaman dan `TARGET`
* `bureau` — riwayat kredit dari lembaga keuangan lain
* `bureau_balance` — riwayat bulanan kredit pada bureau
* `previous_application` — riwayat pengajuan pinjaman sebelumnya di Home Credit
* `POS_CASH_balance` — riwayat bulanan pinjaman POS/Cash
* `credit_card_balance` — riwayat penggunaan kartu kredit
* `installments_payments` — riwayat pembayaran cicilan
* `HomeCredit_columns_description.csv` — deskripsi kolom

Tabel-tabel riwayat diagregasi ke level `SK_ID_CURR` sebelum digabungkan dengan tabel utama.

## Analysis & Data Preparation

### Exploratory Data Analysis

Beberapa kondisi awal yang ditemukan:

* Data tidak seimbang, dengan sekitar **8.07%** nasabah mengalami gagal bayar.
* Terdapat **41 kolom** dengan missing value lebih dari 50%.
* `DAYS_EMPLOYED = 365243` muncul pada sekitar **18.01%** baris dan diperlakukan sebagai anomali.
* `AMT_INCOME_TOTAL` memiliki outlier ekstrem, sehingga dilakukan capping pada persentil 99.5.

Analisis juga mencakup distribusi target, missing value, korelasi fitur numerik, kategori tidak valid, outlier, default rate berdasarkan kategori, serta pengecekan duplikat dan `SK_ID_CURR`.

### Data Cleaning & Feature Engineering

Penanganan data meliputi:

* Menandai anomali `DAYS_EMPLOYED` dengan flag kemudian menggantinya menjadi `NaN`.
* Mengganti kategori `CODE_GENDER = 'XNA'` dengan modus.
* Melakukan capping `AMT_INCOME_TOTAL` pada persentil 99.5.
* Membuat fitur rasio finansial seperti:

  * `CREDIT_INCOME_RATIO`
  * `ANNUITY_INCOME_RATIO`
  * `CREDIT_TERM`
  * `CREDIT_GOODS_RATIO`
  * `EMPLOYED_BIRTH_RATIO`
  * `EXT_SOURCE_MEAN`
* Mengubah beberapa fitur berbasis hari menjadi tahun.
* Mengagregasi data `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, dan `installments_payments` ke level nasabah.
* Menambahkan flag `HAS_*` untuk membedakan nasabah yang memiliki riwayat dengan nasabah yang tidak memiliki riwayat.

Hasil akhirnya berupa satu master dataset dengan satu baris per nasabah yang siap digunakan untuk analisis dan modeling.

## Key Insights

### 1. Previous Credit Behavior

Default rate meningkat seiring meningkatnya proporsi pengajuan sebelumnya yang ditolak:

| Riwayat penolakan | Default rate |
| ----------------- | -----------: |
| 0% ditolak        |        7.07% |
| 1–30% ditolak     |        8.37% |
| 31–60% ditolak    |       11.86% |
| >60% ditolak      |       16.18% |

Pola yang sama terlihat pada keterlambatan pembayaran cicilan:

| Riwayat keterlambatan | Default rate |
| --------------------- | -----------: |
| Tidak pernah telat    |        6.83% |
| 1–10% telat           |        7.92% |
| 11–30% telat          |       10.11% |
| >30% telat            |       13.33% |

Nasabah dengan riwayat penolakan dan keterlambatan yang lebih tinggi menunjukkan risiko gagal bayar yang lebih tinggi.

### 2. Combination of Risk Signals

Kombinasi antara **external score yang rendah** dan **tingginya rasio keterlambatan pembayaran** memberikan pemisahan risiko yang lebih kuat dibandingkan menggunakan satu sinyal saja.

Fitur yang menonjol dari Logistic Regression antara lain `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_MEAN`, `ANNUITY_INCOME_RATIO`, `BUREAU_ACTIVE_FLAG_SUM`, dan `INST_LATE_RATIO`.

## Modeling

Data dibagi secara stratified menjadi:

* **60% train**
* **20% validation**
* **20% test**

One-hot encoding digunakan untuk fitur kategorikal. Imputasi dan scaling untuk Logistic Regression dilakukan di dalam `Pipeline`, sedangkan hyperparameter tuning dilakukan menggunakan **3-fold cross-validation** pada sampel 100.000 baris dari training data. Setelah tuning, model terbaik dari masing-masing algoritma dilatih kembali menggunakan seluruh training data.

### Models

Tiga model dibandingkan:

1. **Logistic Regression**
2. **HistGradientBoosting**
3. **Random Forest**

ROC-AUC digunakan sebagai metric utama untuk tuning dan pemilihan model. `class_weight` dan SMOTE tidak digunakan agar probabilitas model tetap dapat digunakan dalam analisis threshold bisnis.

Hasil tuning menunjukkan:

| Model                | CV ROC-AUC |
| -------------------- | ---------: |
| Logistic Regression  |     0.7631 |
| HistGradientBoosting |     0.7719 |
| Random Forest        |     0.7542 |

**HistGradientBoosting** menjadi model terbaik berdasarkan validation ROC-AUC.

## Model Evaluation

Evaluasi dilakukan menggunakan:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* Confusion Matrix

ROC-AUC digunakan sebagai metric utama karena tidak bergantung pada threshold, sedangkan PR-AUC digunakan untuk melihat performa pada kelas gagal bayar yang merupakan kelas minoritas.

Selain evaluasi performa model, dilakukan analisis feature importance menggunakan:

* Koefisien Logistic Regression
* Permutation importance pada HistGradientBoosting

Threshold juga dievaluasi pada beberapa nilai untuk melihat trade-off antara recall dan precision.

## Business Impact

Analisis bisnis menggunakan dua asumsi:

* **LGD = 45%**
* **Margin = 10%**

Threshold tidak langsung ditetapkan pada 0.50. Threshold dipilih berdasarkan **net outcome pada validation data**, kemudian threshold tersebut diterapkan pada test data untuk mengukur dampaknya.

Logika keputusan:

* Probabilitas gagal bayar di bawah threshold → approve
* Probabilitas gagal bayar di atas atau sama dengan threshold → reject

Net outcome dihitung dari margin pinjaman yang lunas dikurangi kerugian dari pinjaman yang gagal bayar.

Pada hasil akhir, penggunaan model menunjukkan penurunan default rate relatif sebesar **30.6%**, dengan perbedaan net outcome sekitar **200.6 juta** dibandingkan baseline approve-all.

## Risk Tiering Recommendation

Selain keputusan biner approve/reject, nasabah dibagi menjadi **10 risk deciles** berdasarkan probabilitas gagal bayar.

Pendekatan ini digunakan sebagai dasar sistem tiering sehingga keputusan kredit dapat disesuaikan dengan tingkat risiko, misalnya approve standar, review lebih lanjut, atau penolakan/penyesuaian terms untuk kelompok risiko tertinggi.

Sensitivity analysis juga dilakukan terhadap beberapa asumsi LGD dan margin untuk melihat bagaimana perubahan asumsi bisnis memengaruhi threshold optimal.

## Recommendations

* Menggunakan model sebagai **risk tiering system**, bukan hanya keputusan biner.
* Menggunakan threshold berdasarkan analisis cost-benefit, bukan threshold default 0.50.
* Melakukan verifikasi tambahan pada nasabah dengan kombinasi sinyal risiko tinggi.
* Menyesuaikan `principal`, `maturity`, atau repayment terms berdasarkan tingkat risiko.
* Mengganti asumsi LGD dan margin dengan angka aktual ketika model diterapkan pada kondisi bisnis nyata.
* Melakukan monitoring model secara berkala.

Notebook mencakup seluruh proses mulai dari data loading, EDA, data cleaning, feature engineering, aggregation, insight, modeling, evaluation, hingga business impact analysis.

Model terbaik, feature columns, threshold, asumsi LGD/margin, dan versi scikit-learn juga disimpan dalam `model_best.pkl` dan `meta.json` untuk kebutuhan penggunaan kembali model.

## Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
