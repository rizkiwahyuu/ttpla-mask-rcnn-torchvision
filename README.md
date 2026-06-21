# TTPLA Mask R-CNN Torchvision

Implementasi **Mask R-CNN menggunakan Torchvision** untuk tugas **object detection** dan **instance segmentation** pada dataset **TTPLA**. Project ini berfokus pada pendeteksian dan segmentasi objek jaringan transmisi seperti **cable**, **tower_lattice**, **tower_tucohy**, dan **tower_wooden** dari data citra.

Project ini dibuat sebagai baseline eksperimen computer vision. Pipeline yang dibuat mencakup proses training, evaluasi, testing, visualisasi hasil prediksi, dan dokumentasi hasil eksperimen.

---

## Daftar Isi

- [Deskripsi Project](#deskripsi-project)
- [Tujuan Project](#tujuan-project)
- [Dataset](#dataset)
- [Link Dataset dan Model](#link-dataset-dan-model)
- [Model yang Digunakan](#model-yang-digunakan)
- [Struktur Repository](#struktur-repository)
- [Alur Kerja Project](#alur-kerja-project)
- [Konfigurasi Training](#konfigurasi-training)
- [Proses Training](#proses-training)
- [Hasil Training](#hasil-training)
- [Proses Evaluasi](#proses-evaluasi)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Proses Testing](#proses-testing)
- [Hasil Testing](#hasil-testing)
- [Output Project](#output-project)
- [Cara Menjalankan Project](#cara-menjalankan-project)
- [Interpretasi Hasil](#interpretasi-hasil)
- [Keterbatasan Project](#keterbatasan-project)
- [Rekomendasi Pengembangan](#rekomendasi-pengembangan)
- [Kesimpulan](#kesimpulan)
- [Lisensi](#lisensi)
- [Author](#author)

---

## Deskripsi Project

Project ini membangun pipeline deep learning untuk mendeteksi dan melakukan segmentasi objek pada citra jaringan transmisi listrik. Model yang digunakan adalah **Mask R-CNN ResNet-50 FPN** dari Torchvision.

Mask R-CNN dipilih karena dapat menghasilkan dua jenis output dalam satu model:

1. **Bounding box** untuk menunjukkan lokasi objek.
2. **Segmentation mask** untuk menunjukkan area piksel objek secara lebih detail.

Project ini memiliki tiga notebook utama:

| Notebook | Fungsi |
|---|---|
| Trainer | Melatih model Mask R-CNN pada dataset TTPLA |
| Evaluator | Mengukur performa model menggunakan metrik COCO |
| Tester | Menjalankan inference dan menyimpan hasil prediksi |

Project ini cocok digunakan sebagai baseline awal untuk eksperimen object detection dan instance segmentation berbasis PyTorch.

---

## Tujuan Project

Tujuan utama project ini adalah membuat model baseline Mask R-CNN untuk mendeteksi dan melakukan segmentasi objek pada dataset TTPLA.

Tujuan spesifik project:

- Melatih model Mask R-CNN pada dataset TTPLA.
- Mendeteksi objek cable dan tower dari citra transmisi.
- Menghasilkan bounding box dan segmentation mask.
- Mengevaluasi model menggunakan metrik COCO.
- Menyimpan hasil evaluasi dalam format JSON dan CSV.
- Menampilkan contoh gambar hasil prediksi.
- Menyediakan dokumentasi eksperimen yang rapi untuk GitHub.
- Menjadi dasar pengembangan model computer vision berikutnya.

---

## Dataset

Dataset yang digunakan adalah **TTPLA dataset** dengan format anotasi **COCO JSON**. Dataset ini berisi citra jaringan transmisi listrik dengan anotasi objek berupa bounding box dan segmentation mask.

### Jumlah Data

| Split | Jumlah Gambar |
|---|---:|
| Training | 842 |
| Testing | 400 |

### Class Objek

Model dilatih untuk mengenali 4 class objek utama.

| Label Model | Nama Class |
|---:|---|
| 1 | cable |
| 2 | tower_lattice |
| 3 | tower_tucohy |
| 4 | tower_wooden |

Mask R-CNN juga membutuhkan 1 class tambahan untuk background.

```text
4 class objek + 1 background = 5 class
```

### Format Dataset

Dataset menggunakan format COCO. Format ini terdiri dari beberapa bagian utama.

| Komponen | Keterangan |
|---|---|
| images | Informasi file gambar |
| annotations | Informasi bounding box, segmentation mask, dan class |
| categories | Daftar class objek |

Format COCO dipakai karena sesuai dengan kebutuhan evaluasi object detection dan instance segmentation.

---

## Link Dataset dan Model

File dataset dan model tidak dimasukkan langsung ke repository karena ukurannya besar. File besar lebih baik disimpan di Google Drive atau layanan penyimpanan eksternal.

| File | Link |
|---|---|
| Dataset TTPLA | [Google Drive Dataset](https://drive.google.com/drive/folders/1fE4vunwUvSC0Gf-1plinN3Z821yTYUZk?usp=drive_link) |
| Best Model | [Google Drive Best Model](https://drive.google.com/file/d/1l9fM6ZXW37M3tzUIzGLSYkziFq0VZZzF/view?usp=drive_link) |

Setelah dataset diunduh, letakkan dataset dengan struktur berikut:

```text
data/ttpla-dataset/
├── trainingset/
│   ├── train.json
│   └── image files
│
└── testset/
    ├── test.json
    └── image files
```

Setelah best model diunduh, letakkan file model pada path berikut atau sesuaikan dengan notebook:

```text
outputs/best_model.pth
```

---

## Model yang Digunakan

Model yang digunakan adalah:

```text
Mask R-CNN ResNet-50 FPN
```

Model berasal dari library **Torchvision**.

### Komponen Model

| Komponen | Fungsi |
|---|---|
| ResNet-50 | Backbone untuk ekstraksi fitur gambar |
| FPN | Mengambil fitur dari beberapa skala objek |
| RPN | Mengusulkan kandidat area objek |
| Fast R-CNN Head | Menghasilkan class dan bounding box |
| Mask Head | Menghasilkan segmentation mask |

### Alasan Menggunakan Mask R-CNN

Mask R-CNN cocok untuk project ini karena:

- Dapat mendeteksi posisi objek dengan bounding box.
- Dapat menghasilkan segmentation mask.
- Mendukung format anotasi COCO.
- Tersedia pretrained weight dari COCO.
- Mudah digunakan dengan PyTorch dan Torchvision.

### Penyesuaian Model

Model bawaan COCO memiliki banyak class. Karena project ini hanya menggunakan 4 class objek, bagian head model diganti.

Bagian yang diganti:

- `FastRCNNPredictor` untuk klasifikasi dan bounding box.
- `MaskRCNNPredictor` untuk segmentation mask.

Jumlah class akhir:

```text
num_classes = 5
```

---

## Struktur Repository

Struktur repository yang digunakan:

```text
ttpla-mask-rcnn-torchvision/
├── README.md
├── .gitignore
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   ├── 01_trainer_mask_rcnn_ttpla.ipynb
│   ├── 02_evaluator_mask_rcnn_ttpla.ipynb
│   └── 03_tester_mask_rcnn_ttpla.ipynb
│
├── configs/
│   └── config.yaml
│
├── docs/
│   └── experiment_summary.md
│
├── outputs/
│   ├── metrics/
│   │   ├── training_history.json
│   │   ├── evaluation_metrics.json
│   │   └── tester_summary.json
│   │
│   ├── figures/
│   │   ├── trainer_loss_curve.png
│   │   ├── trainer_map_curve.png
│   │   ├── evaluator_metric_comparison.png
│   │   ├── evaluator_ap_by_object_size.png
│   │   ├── tester_class_prediction_count.png
│   │   └── tester_mean_confidence_by_class.png
│   │
│   └── samples/
│       ├── trainer_sample_prediction.png
│       ├── evaluator_sample_predictions_grid.png
│       ├── tester_sample_predictions_grid_01.png
│       └── tester_sample_predictions_grid_02.png
│
└── results/
    ├── training_history.csv
    ├── evaluation_metrics.csv
    ├── class_prediction_stats.csv
    └── README_results.md
```

### Penjelasan Folder

| Folder | Fungsi |
|---|---|
| `notebooks/` | Berisi notebook trainer, evaluator, dan tester |
| `configs/` | Berisi konfigurasi eksperimen |
| `docs/` | Berisi ringkasan eksperimen |
| `outputs/metrics/` | Berisi hasil metrik dalam format JSON |
| `outputs/figures/` | Berisi grafik training, evaluasi, dan testing |
| `outputs/samples/` | Berisi contoh gambar hasil prediksi |
| `results/` | Berisi hasil eksperimen dalam bentuk CSV dan dokumentasi hasil |

---

## Alur Kerja Project

Pipeline project berjalan dari dataset sampai hasil prediksi.

```text
Dataset TTPLA
↓
Preprocessing anotasi COCO
↓
Training Mask R-CNN
↓
Simpan best_model.pth
↓
Evaluasi menggunakan COCO metric
↓
Testing pada data test
↓
Export hasil prediksi
↓
Visualisasi hasil prediksi
```

### Alur Notebook

| Notebook | Fungsi Utama | Output Utama |
|---|---|---|
| Trainer | Melatih model | `best_model.pth`, `metrics.json`, grafik training |
| Evaluator | Mengukur performa model | `evaluation_metrics.json`, hasil AP dan AR |
| Tester | Menjalankan inference | `predictions.csv`, `predictions.json`, gambar prediksi |

---

## Konfigurasi Training

Konfigurasi utama training:

| Parameter | Nilai |
|---|---:|
| Batch size | 4 |
| Epoch | 15 |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Evaluation interval | Setiap epoch |
| Score threshold | 0.50 |
| Mask threshold | 0.50 |
| Device | CUDA GPU |

### Fungsi Parameter

| Parameter | Penjelasan |
|---|---|
| Batch size | Jumlah gambar yang diproses dalam satu iterasi |
| Epoch | Jumlah putaran training seluruh dataset |
| Learning rate | Kecepatan model dalam memperbarui bobot |
| Momentum | Membantu optimizer bergerak lebih stabil |
| Weight decay | Mengurangi risiko overfitting |
| Score threshold | Batas confidence minimum untuk prediksi |
| Mask threshold | Batas piksel mask agar dianggap objek |

---

## Proses Training

Training dilakukan dengan tahapan berikut:

```text
Load dataset
↓
Baca anotasi COCO
↓
Konversi anotasi ke format PyTorch
↓
Buat Dataset dan DataLoader
↓
Load model Mask R-CNN pretrained COCO
↓
Ganti head model sesuai jumlah class
↓
Training selama 15 epoch
↓
Evaluasi setiap epoch
↓
Simpan model terbaik
↓
Simpan metrik training
```

### Augmentasi Data

Augmentasi digunakan untuk membantu model lebih tahan terhadap variasi gambar.

| Augmentasi | Fungsi |
|---|---|
| Horizontal flip | Melatih model agar tidak bergantung pada arah horizontal objek |
| Vertical flip | Menambah variasi posisi objek |
| Random brightness contrast | Membuat model lebih tahan terhadap variasi pencahayaan |
| RGB shift | Membuat model lebih tahan terhadap perubahan warna |
| Blur | Membuat model lebih tahan terhadap gambar kurang tajam |

---

## Hasil Training

Model dilatih selama 15 epoch.

Ringkasan hasil training:

| Metric | Nilai |
|---|---:|
| Final train loss | 0.7378 |
| Best epoch | 15 |
| Final BBox AP | 0.1915 |
| Final Segmentation AP | 0.1235 |

### Interpretasi Hasil Training

Nilai loss menurun selama proses training. Hal ini menunjukkan bahwa model berhasil belajar dari data.

Namun, nilai AP masih tergolong rendah. Artinya, model sudah dapat mengenali pola objek, tetapi akurasinya masih perlu ditingkatkan.

Hasil training ini dapat dianggap sebagai baseline awal, bukan hasil final untuk deployment produksi.

---

## Proses Evaluasi

Evaluasi dilakukan menggunakan metrik COCO. Evaluator memakai file model terbaik.

```text
best_model.pth
```

Konfigurasi evaluasi:

| Setting | Nilai |
|---|---:|
| Jumlah gambar test | 400 |
| Checkpoint | best_model.pth |
| Checkpoint epoch | 15 |
| Score threshold | 0.05 |
| Mask threshold | 0.50 |
| Jumlah prediksi bbox | 17,956 |
| Jumlah prediksi segmentation | 17,956 |
| Waktu evaluasi | 2,204.94 detik |

### Mengapa Evaluasi Memakai Score Threshold 0.05?

Evaluasi COCO biasanya menggunakan threshold rendah agar lebih banyak kandidat prediksi dihitung. Dengan cara ini, AP dapat dihitung secara lebih menyeluruh pada berbagai tingkat confidence.

---

## Hasil Evaluasi

Hasil evaluasi utama:

| Metric | BBox | Segmentation |
|---|---:|---:|
| AP | 0.2146 | 0.1214 |
| AP50 | 0.3602 | 0.2195 |
| AP75 | 0.2342 | 0.1254 |
| AP Small | 0.1022 | 0.0008 |
| AP Medium | 0.2849 | 0.0625 |
| AP Large | 0.2258 | 0.2416 |
| AR 1 | 0.1706 | 0.1503 |
| AR 10 | 0.3337 | 0.2106 |
| AR 100 | 0.3594 | 0.2148 |

### Penjelasan Metrik

| Metrik | Penjelasan |
|---|---|
| AP | Average Precision pada beberapa threshold IoU |
| AP50 | AP saat IoU minimal 0.50 |
| AP75 | AP saat IoU minimal 0.75 |
| AP Small | AP untuk objek kecil |
| AP Medium | AP untuk objek sedang |
| AP Large | AP untuk objek besar |
| AR | Average Recall, yaitu kemampuan model menemukan objek |

### Interpretasi Evaluasi

Hasil evaluasi menunjukkan:

- BBox AP lebih tinggi daripada Segmentation AP.
- Model lebih baik dalam mendeteksi lokasi objek dibanding menggambar area mask.
- Segmentasi objek kecil sangat lemah.
- Segmentasi objek besar lebih baik dibanding objek kecil.
- Model masih membutuhkan peningkatan kualitas data dan tuning parameter.

---

## Proses Testing

Testing dilakukan untuk menjalankan inference pada semua gambar test.

Konfigurasi testing:

| Setting | Nilai |
|---|---:|
| Jumlah gambar diproses | 400 |
| Score threshold | 0.30 |
| Mask threshold | 0.50 |
| Total prediksi | 6,064 |
| Rata-rata prediksi per gambar | 15.16 |
| Waktu testing | 2,867.63 detik |

Testing menghasilkan:

- Prediksi bounding box.
- Prediksi class.
- Confidence score.
- Prediksi segmentation mask.
- File CSV.
- File JSON.
- Gambar visualisasi hasil prediksi.

---

## Hasil Testing

Statistik prediksi per class:

| Class | Total Prediksi | Mean Score | Max Score | Mean Area |
|---|---:|---:|---:|---:|
| cable | 5,815 | 0.6793 | 0.9992 | 40,277.92 |
| tower_lattice | 249 | 0.6073 | 0.9673 | 76,941.33 |

### Interpretasi Testing

Hasil testing menunjukkan bahwa model paling sering memprediksi class **cable**. Prediksi class **tower_lattice** muncul dalam jumlah lebih sedikit.

Pada threshold 0.30, class berikut belum muncul dalam hasil testing:

```text
tower_tucohy
tower_wooden
```

Kemungkinan penyebab:

- Data tiap class tidak seimbang.
- Class tower_tucohy dan tower_wooden memiliki jumlah sampel lebih sedikit.
- Bentuk antar jenis tower mirip.
- Model masih belum cukup kuat membedakan class tower.
- Threshold 0.30 terlalu tinggi untuk class yang confidence-nya rendah.

---

## Output Project

### Output Training

```text
best_model.pth
last_model.pth
metrics.json
final_evaluation.json
loss_curve.png
map_curve.png
sample_prediction.png
```

### Output Evaluasi

```text
evaluation_metrics.json
loss_curve_from_training_history.png
map_curve_from_training_history.png
sample_predictions_summary.json
sample prediction images
```

### Output Testing

```text
predictions.json
predictions.csv
tester_summary.json
class_prediction_stats.csv
test prediction images
```

---

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/rizkiwahyuu/ttpla-mask-rcnn-torchvision.git
cd ttpla-mask-rcnn-torchvision
```

### 2. Install Dependency

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download dataset melalui link berikut:

[Download Dataset TTPLA](https://drive.google.com/drive/folders/1fE4vunwUvSC0Gf-1plinN3Z821yTYUZk?usp=drive_link)

Letakkan dataset sesuai struktur berikut:

```text
data/ttpla-dataset/
├── trainingset/
│   ├── train.json
│   └── image files
│
└── testset/
    ├── test.json
    └── image files
```

### 4. Download Best Model

Download best model melalui link berikut:

[Download Best Model](https://drive.google.com/file/d/1l9fM6ZXW37M3tzUIzGLSYkziFq0VZZzF/view?usp=drive_link)

Letakkan file model pada path berikut atau sesuaikan dengan notebook:

```text
outputs/best_model.pth
```

### 5. Jalankan Notebook

Jalankan notebook sesuai urutan berikut:

```text
1. notebooks/01_trainer_mask_rcnn_ttpla.ipynb
2. notebooks/02_evaluator_mask_rcnn_ttpla.ipynb
3. notebooks/03_tester_mask_rcnn_ttpla.ipynb
```

### Catatan Penggunaan

Jika hanya ingin melakukan evaluasi atau testing, Anda tidak perlu menjalankan trainer dari awal. Cukup download `best_model.pth`, lalu jalankan notebook evaluator atau tester.

---

## Interpretasi Hasil

Model berhasil menjalankan pipeline dari training sampai testing. Namun performa model masih berada pada level baseline.

Ringkasan interpretasi:

| Bagian | Kondisi |
|---|---|
| Training | Model berhasil belajar, loss menurun |
| BBox Detection | Lebih baik daripada segmentasi |
| Segmentation | Masih rendah |
| Objek kecil | Sangat sulit dikenali |
| Prediksi class | Didominasi oleh cable |
| Kesiapan produksi | Belum siap |
| Kesiapan riset lanjutan | Sudah layak sebagai baseline |

Hasil ini menunjukkan bahwa model sudah dapat mengenali beberapa objek, tetapi belum cukup stabil untuk penggunaan nyata tanpa pengembangan lanjutan.

---

## Keterbatasan Project

Project ini memiliki beberapa keterbatasan:

1. Training hanya dilakukan selama 15 epoch.
2. Distribusi data antar class kemungkinan belum seimbang.
3. Model masih lemah pada objek kecil.
4. Segmentasi mask belum presisi.
5. Beberapa class tower belum muncul pada hasil testing.
6. Threshold testing memengaruhi jumlah prediksi.
7. Model belum diuji pada data lapangan yang benar-benar baru.
8. Hasil AP masih rendah untuk deployment produksi.

---

## Rekomendasi Pengembangan

Rekomendasi pengembangan berikutnya:

1. Menambah jumlah data untuk class minoritas.
2. Melakukan balancing data antar class.
3. Memperbaiki kualitas anotasi segmentation mask.
4. Melatih model dengan epoch lebih banyak.
5. Melakukan hyperparameter tuning.
6. Menguji beberapa nilai score threshold.
7. Menggunakan backbone yang lebih kuat.
8. Melakukan evaluasi AP per class.
9. Membandingkan performa dengan YOLO untuk object detection.
10. Menguji model pada gambar lapangan yang baru.
11. Membuat aplikasi inference sederhana berbasis web.
12. Mengintegrasikan hasil model dengan dashboard monitoring aset.

---

## Kesimpulan

Project ini berhasil membangun pipeline lengkap Mask R-CNN Torchvision untuk dataset TTPLA. Pipeline mencakup proses training, evaluasi, testing, ekspor hasil prediksi, dan visualisasi.

Hasil evaluasi menunjukkan bahwa model sudah mampu mendeteksi sebagian objek pada citra transmisi. Performa bounding box lebih baik dibanding segmentation mask. Namun, performa segmentasi masih rendah, terutama pada objek kecil dan class minoritas.

Dengan demikian, project ini layak digunakan sebagai baseline awal untuk eksperimen computer vision berbasis object detection dan instance segmentation. Untuk penggunaan produksi, model masih membutuhkan peningkatan kualitas dataset, tuning parameter, dan pengujian lanjutan pada data lapangan.

---

## Lisensi

Project ini menggunakan lisensi MIT. Lihat file `LICENSE` untuk informasi lengkap.

---

## Author

**Rizki Wahyu Widodo**

Project ini dibuat untuk dokumentasi eksperimen computer vision menggunakan Mask R-CNN Torchvision pada dataset TTPLA.
