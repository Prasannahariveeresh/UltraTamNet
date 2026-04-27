# Data Availability

## Statement for Manuscript

> **Dataset 1 (uTHCD — public benchmark):** The Unconstrained Tamil Handwritten Character
> Database (uTHCD) used in this study is publicly available on Kaggle at
> https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database
> (156 classes, ≈91,000 images).
>
> **Dataset 2 (Custom Tamil Vowel Dataset — authors' dataset):** The custom handwritten Tamil
> vowel dataset collected for this study is publicly available on Kaggle at
> https://www.kaggle.com/datasets/prasanna2209/ultratamnet-custom-dataset/
> (12 Tamil vowel classes, 484 raw images).
>
> **Supplementary data supporting Tables III and VI** (epoch-by-epoch training logs and
> final metrics from all trained models) are provided as Supplementary Material in
> `supplementary/`.

---

## Dataset Citations

### Dataset 1 — uTHCD

```
Shaffi, N. and Hajamohideen, F. (2021).
uTHCD: a new benchmarking for Tamil handwritten OCR.
IEEE Access, 9, pp.101469–101493.
https://doi.org/10.1109/ACCESS.2021.3097444

Dataset available at:
https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database
```

### Dataset 2 — Custom Tamil Vowel Dataset

```
Megalingam, R.K. and Jeyaveerapandian Raji, P. (2025).
Custom Handwritten Tamil Vowel Dataset (Uyir Ezhuthu, 12 classes).
Collected at HuT Labs, Amrita Vishwa Vidyapeetham, Amritapuri, India.
484 raw images across 12 Tamil vowel classes.
Available at: https://www.kaggle.com/datasets/prasanna2209/ultratamnet-custom-dataset/
```

---

## Data Supporting Tables III and VI

The raw training logs (epoch-by-epoch train/validation accuracy and loss) for all
experiments reported in Tables III and VI are provided in:

```
supplementary/
├── table3_raw_results.csv          — final test metrics for all 12 models (Table III)
├── table6_raw_results.csv          — final test metrics for all augmentation runs (Table VI)
├── training_logs/
│   ├── table3/                     — per-model epoch-by-epoch training history CSVs
│   └── table6/                     — per-model-per-augmentation training history CSVs
└── README.md                       — description of supplementary contents
```

These logs were generated during training on an NVIDIA GeForce RTX 3060 (12 GB VRAM)
as described in Section V-B of the paper.
