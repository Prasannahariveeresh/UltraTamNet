# Code Availability

## Statement for Manuscript

> The custom code used to design, train, and evaluate UltraTamNet is publicly available at
> **[GitHub — Prasannahariveeresh/UltraTamNet](https://github.com/Prasannahariveeresh/UltraTamNet)**
> (DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX), archived via Zenodo).
> The repository contains all source code required to reproduce Tables III, IV, VI and
> Figures 9–16, including the UltraTamNet model architecture, all 11 baseline model
> builders, dataset preprocessing pipelines, augmentation utilities, and evaluation scripts.

---

## What the Repository Contains

| Component | File(s) | Reproduces |
|-----------|---------|-----------|
| UltraTamNet architecture | `models/ultratamnet.py` | Table II, core claim |
| Baseline model builders | `models/baselines.py` | Table III |
| uTHCD preprocessing | `data/preprocess_uthcd.py` | Dataset 1 loading |
| Custom dataset preprocessing | `data/preprocess_custom.py` | Dataset 2 loading |
| Offline augmentation pipeline | `augmentation/augment_custom_dataset.py` | Table VI |
| Table III training script | `experiments/train_uthcd_benchmark.py` | Table III |
| Table VI training script | `experiments/train_augmentation_study.py` | Table VI |
| Ablation study | `experiments/ablation_study.py` | Table IV |
| Metrics & visualisation | `utils/evaluate.py` | Figs 9–16 |
| Figure generation scripts | `figures/plot_*.py` | Figs 7–16 |

## Software Environment

| Dependency | Version |
|-----------|---------|
| Python | 3.12.4 |
| TensorFlow / Keras | ≥ 2.12.0 |
| NumPy | 1.23.5 (pinned for imgaug compatibility) |
| OpenCV | ≥ 4.7 |
| scikit-learn | ≥ 1.2 |
| imgaug | 0.4.0 |
| pandas | ≥ 1.5 |
| matplotlib | ≥ 3.6 |

Full pinned dependencies: `requirements.txt`

## Hardware Used in Paper

- CPU: Intel Xeon E5-2680
- GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- RAM: 32 GB
- OS: Ubuntu 24.04 LTS
- Framework: TensorFlow 2.x with CUDA

## How to Archive a Snapshot (GitHub → Zenodo)

1. Push this repository to GitHub.
2. Log in to [zenodo.org](https://zenodo.org) with your GitHub account.
3. Enable the repository under **Zenodo → GitHub → Linked repositories**.
4. Create a GitHub Release (e.g. `v1.0.0`); Zenodo will automatically mint a DOI.
5. Replace `10.5281/zenodo.XXXXXXX` above with the issued DOI.

See: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
