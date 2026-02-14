# Multi-Image Panorama Stitching using SIFT

Project ini merupakan implementasi **Panorama Stitching** menggunakan algoritma  
**SIFT (Scale-Invariant Feature Transform)** untuk menggabungkan **3 citra** menjadi panorama.

---

## Dataset

Letakkan 3 gambar pada folder berikut:
```bash
images/
├── gambar1.jpg
├── gambar2.jpg
└── gambar3.jpg
```


Gambar harus memiliki area overlap agar stitching berhasil.

---

## Requirements

Install library yang dibutuhkan:

```bash
pip install opencv-python numpy matplotlib
```
Cara Menjalankan
Project terdiri dari 6 tahapan utama:
```bash
| Step |            Proses            |             Output File             |
|------|------------------------------|-------------------------------------|
| 01   | Scale-space & gradien        | 01_space_feature_analysis.png       |  
| 02   | Difference of Gaussian (DoG) | 02_dog_blobs.png                    |
| 03   | Deteksi Keypoints SIFT       | 03_visualisasi_proses_keypoints.png |
| 04   | Visualisasi Descriptor       | 04_descriptor_vector.png            |
| 05   | Feature Matching + RANSAC    | 05_robust_matching_3img.png         |
| 06   | Panorama Stitching 3 Gambar  | 06_multi_panorama_steps.png         |

```
Run masing-masing file step:
```bash
python step01.py
python step02.py
...
python step06.py
