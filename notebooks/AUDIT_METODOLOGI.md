# Audit Metodologi Paper dan Desain Koreksi Replikasi

Dokumen kerja untuk Tugas #2 bagian 3 (Replicate and Correct the Paper's Methodology). Isi di sini
nanti dipindahkan/diringkas ke bagian 3.1-3.2 laporan (template.docx).

Sumber: `Ringkasan_Paper.md` (ringkasan Tugas #1), karena paper PDF asli tidak dibaca ulang untuk tugas
ini (disepakati dengan user).

---

## 1. Pipeline yang Dilaporkan Paper (Reported Experimental Pipeline)

| Komponen Pipeline | Metode yang Dilaporkan | Detail/Parameter (dari Ringkasan_Paper.md) |
|---|---|---|
| Data preparation | Load dataset UCI Online Shoppers Purchasing Intention | 12.330 sesi, 18 kolom (10 numerik, 7 kategorikal, 1 label `Revenue`) |
| Preprocessing | (a) One-hot encoding pada fitur kategorikal string; (b) Standard scaling pada fitur numerik | (a) Hanya `Month` dan `VisitorType` yang di-one-hot (dikonfirmasi lewat rekonstruksi jumlah fitur 28 di notebook 02_eda); fitur kode-integer (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) dan biner (`Weekend`) dipakai apa adanya. (b) Scaling: mean 0, std 1, untuk fitur numerik. **Urutan scaling vs split TIDAK disebutkan eksplisit di ringkasan.** |
| Feature engineering/selection | Seleksi fitur Chi-square (χ²) | Dari 28 fitur (setelah encoding) diseleksi top-k, k diuji {10, 15, 20}, hasil terbaik k=20. **Urutan seleksi vs split TIDAK disebutkan eksplisit.** |
| Data split / validation | Train/test split | 70% train, 30% test. Metode split (stratified/random, random_state) tidak disebutkan. |
| Imbalance handling | SMOTE (oversampling) dan random undersampling | Ringkasan Tugas #1 (baris 67 dan Q6) secara EKSPLISIT menyatakan: "Penanganan imbalance data dilakukan dengan SMOTE (oversampling) dan random undersampling, yang hanya diterapkan pada data training (bukan data test)". Ini satu-satunya komponen preprocessing yang urutannya (setelah split) dinyatakan jelas oleh paper sendiri. |
| Model(s) | DT, SVM, MLP, RF, XGBoost | DT: entropy, max_depth {5,10,15}, terbaik 5. SVM: RBF, C=7. MLP: 3 hidden layer (28,56,28), SGD, lr=0,00005, ReLU. RF: max_depth=20, n_estimators=100. XGBoost: max_depth {2,5,10}, terbaik 2. |
| Evaluation | Accuracy, Precision, Recall/TPR, F1, TNR, MCC, auROC, auPR | Dihitung pada test set (30%). Eksperimen dilakukan bertahap (ablation study): baseline, +feature selection, +oversampling, +undersampling, kombinasi. |

---

## 2. Isu, Ambiguitas, dan Keputusan Replikasi

| # | Yang Dilaporkan Paper | Isu / Informasi yang Hilang | Keputusan Replikasi | Justifikasi |
|---|---|---|---|---|
| 1 | Standard scaling pada fitur numerik, urutan vs split tidak disebutkan | **Kandidat leakage utama.** Jika scaler (mean, std) di-fit pada seluruh dataset (train+test) sebelum split, maka statistik dari data test "mengintip" ke proses training (nilai test ikut menentukan parameter transformasi yang dipakai untuk men-scale data train). Ini adalah salah satu contoh persis yang disebutkan brief tugas ("if the authors scaled ... using the entire dataset before the train/test split"). | Split dilakukan LEBIH DULU (70/30, stratified by Revenue). `StandardScaler` di-`fit` HANYA pada data train, lalu `transform` diterapkan ke train dan test menggunakan parameter yang sama (fit dari train). | Mencegah informasi distribusi test bocor ke proses scaling train, sesuai praktik standar scikit-learn (`fit` di train, `transform` di train & test) dan sesuai instruksi brief. |
| 2 | Seleksi fitur χ² (top-20 dari 28), urutan vs split tidak disebutkan | **Kandidat leakage kedua.** χ² dihitung berdasarkan hubungan fitur-label. Jika dihitung pada seluruh dataset sebelum split, keputusan "fitur mana yang signifikan" ikut ditentukan oleh label test set, sehingga model secara tidak langsung "melihat" test set saat memilih fitur yang dipakai. | χ² (`SelectKBest`) di-`fit` HANYA pada data train (X_train, y_train) untuk menentukan top-20 fitur, lalu daftar fitur yang sama (bukan nilai χ² baru) diterapkan ke test set. | Mencegah label test memengaruhi pemilihan fitur; konsisten dengan aturan brief "use the test set only for final evaluation, not for preprocessing decisions". |
| 3 | χ² butuh nilai non-negatif, tapi standard scaling menghasilkan nilai bisa negatif (mean 0) | Paper tidak menjelaskan bagaimana χ² tetap bisa dihitung pada fitur yang sudah di-scale (yang bisa bernilai negatif) — kemungkinan χ² dihitung SEBELUM scaling (pada data numerik asli + fitur biner/one-hot yang non-negatif), baru scaling diterapkan setelah fitur terpilih. Ini adalah gap informasi di paper. | Urutan diasumsikan: (1) split, (2) encoding (one-hot Month & VisitorType), (3) χ² dihitung pada fitur HASIL ENCODING tapi SEBELUM scaling (semua non-negatif secara alami: fitur numerik asli, dummy 0/1, kode integer), fit di train saja, (4) scaling standard HANYA pada fitur numerik yang lolos seleksi, fit di train, transform ke train+test. | Asumsi ini didokumentasikan secara eksplisit sesuai instruksi brief ("if important methodological information is missing... make a reasonable decision and clearly document your assumption"). Urutan ini secara teknis valid (χ² tidak butuh scaling) dan tetap mendekati skema paper. |
| 4 | SMOTE/undersampling hanya di train (dinyatakan eksplisit oleh paper) | Tidak ada isu leakage di sini — paper SUDAH benar. Tapi perlu dipastikan implementasi replikasi juga demikian. | SMOTE (`imblearn`) di-`fit_resample` HANYA pada X_train/y_train setelah split dan setelah scaling+seleksi fitur. Test set TIDAK di-resample, tetap mencerminkan distribusi asli (imbalanced). | Menjaga konsistensi dengan yang secara eksplisit dinyatakan benar oleh paper; tidak ada koreksi diperlukan di komponen ini, hanya verifikasi implementasi. |
| 5 | Metode split (random_state, stratified atau tidak) tidak disebutkan | Tanpa random_state, hasil replikasi tidak reproducible dan sulit dibandingkan run-to-run. | Split dilakukan dengan `train_test_split(..., test_size=0.3, stratify=y, random_state=42)`. | random_state=42 dipilih sebagai konvensi umum agar reproducible; stratify=y dipilih karena data sangat imbalanced, memastikan proporsi kelas di train dan test tetap representatif (sesuai instruksi umum praktik data splitting yang valid untuk classification imbalanced, meskipun paper tidak menyebutkan strata secara eksplisit). |
| 6 | Ablation study paper: baseline, +feature selection, +oversampling, +undersampling, +kombinasi (χ²+SMOTE terbaik untuk XGBoost) | Brief tugas #2 melarang penambahan preprocessing/model/tuning baru untuk sekadar meningkatkan performa (perbaikan disimpan untuk Tugas #3). | Replikasi difokuskan pada skenario **kombinasi terbaik yang dilaporkan paper** (χ² top-20 + SMOTE) untuk kelima model, BUKAN mengulang seluruh 6 skenario ablation. Hasil utama yang dibandingkan adalah XGBoost dan Decision Tree pada skenario ini (Tabel hasil utama paper, baris 85-86 Ringkasan_Paper.md). | Menjaga eksperimen tetap dekat dengan hasil utama yang dilaporkan paper dan yang akan dibandingkan di Bagian 4 laporan, tanpa memperluas scope ke eksperimen tambahan yang dilarang brief untuk Tugas #2. |
| 7 | Hyperparameter model (lihat tabel Bagian 1) | Paper hanya melaporkan nilai hyperparameter TERBAIK, bukan proses tuning/CV yang detail (mis. apakah grid search dilakukan di seluruh data atau hanya train). | Hyperparameter yang dilaporkan sebagai "terbaik" oleh paper dipakai LANGSUNG (tidak dilakukan tuning ulang), sesuai instruksi brief ("do not introduce ... extensive hyperparameter tuning simply to improve performance"). | Sesuai instruksi eksplisit brief; juga karena tuning ulang butuh akses ke proses CV paper yang tidak didokumentasikan. |

---

## 3. Ringkasan Koreksi Utama (untuk Bagian 3.2 Laporan)

**Koreksi #1 -- Scaling di-fit setelah split, bukan sebelum:**
- Yang paper lakukan/laporkan: standard scaling pada fitur numerik, urutan terhadap split tidak dijelaskan.
- Masalah: jika di-fit pada seluruh dataset, parameter scaling (mean, std) ikut ditentukan oleh data test, menyebabkan data leakage ringan yang bisa membuat evaluasi performa optimistis secara tidak wajar.
- Yang diubah: scaler di-fit HANYA pada data train, ditransformasikan ke train dan test.
- Kenapa lebih tepat: memastikan test set benar-benar "unseen" sampai tahap evaluasi akhir, sesuai praktik standar dan instruksi brief.

**Koreksi #2 -- Seleksi fitur χ² di-fit setelah split, bukan sebelum:**
- Yang paper lakukan/laporkan: χ² dipakai untuk memilih top-20 dari 28 fitur, urutan terhadap split tidak dijelaskan.
- Masalah: jika χ² dihitung dari seluruh dataset (termasuk label test), keputusan fitur mana yang dipakai model sudah "mengintip" pola di data test, sehingga skema seleksi fitur menjadi bias terhadap performa test set tersebut.
- Yang diubah: χ² dihitung dan fitur top-20 ditentukan HANYA dari data train, daftar fitur yang sama diterapkan pada test.
- Kenapa lebih tepat: mencegah keputusan preprocessing/feature engineering "melihat" data evaluasi, sesuai instruksi brief bahwa test set hanya untuk evaluasi akhir.

**Asumsi yang didokumentasikan (bukan koreksi, tapi keputusan atas informasi yang hilang):**
- Urutan χ² dilakukan sebelum scaling (karena χ² butuh nilai non-negatif).
- Split menggunakan `random_state=42`, stratified by target.
- Fokus replikasi pada skenario kombinasi terbaik paper (χ²+SMOTE), bukan seluruh 6 skenario ablation.

---

## 4. Pipeline Final yang Akan Diimplementasikan (Task 5-6)

```
Data (12.330 x 18)
  -> Train/Test Split (70/30, stratified, random_state=42)
       -> [FIT hanya di train, transform ke train+test:]
  -> One-hot encoding (Month, VisitorType)
  -> Seleksi fitur Chi-square, top-20 dari 28 (fit di train)
  -> Standard scaling fitur numerik yang lolos seleksi (fit di train)
       -> [Hanya di train:]
  -> SMOTE oversampling (train saja; test tetap asli/imbalanced)
  -> Model: DT, SVM, MLP, RF, XGBoost (parameter sesuai paper)
  -> Evaluasi pada test set: Accuracy, Precision, TPR, F1, TNR, MCC, auROC, auPR
```

Perbedaan dengan asumsi pipeline paper (yang tidak menyebutkan urutan split vs preprocessing secara
eksplisit): pada replikasi ini, SPLIT dilakukan di awal, sebelum semua langkah preprocessing yang
melibatkan estimasi parameter dari data (scaling, seleksi fitur, resampling).
