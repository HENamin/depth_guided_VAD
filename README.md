# CMP_3CNN_VAD: Depth-Guided Video Anomaly Detection Using Compact Representation of Spatio-Temporal Features

## Overview
This repository contains the official PyTorch implementation of the paper **"Depth-Guided Video Anomaly Detection Using Compact Representation of Spatio-Temporal Features"** (Habib Ebadi Namin, Reza Kharghanian, and Alireza Ahmadyfard).

The proposed framework presents an unsupervised U-Net-based spatiotemporal autoencoder for video anomaly detection in surveillance applications. The architecture incorporates a residual 2D spatial encoder (`Res2D`) alongside a hierarchical 3D convolutional bottleneck (`Res3D`) to effectively capture spatial structures and temporal continuity across video clips. To structure the latent representation space during training on normal video sequences, an EWMA-smoothed channel-wise feature compactness module is incorporated into the bottleneck. During evaluation, a depth-guided anomaly scoring mechanism is applied using monocular depth estimation maps to compensate for perspective distortions and distance-dependent motion scale variations.

---

## Experimental Results

Best frame-level Area Under the ROC Curve (AUC-ROC %) performance comparing models **Without Depth** vs. **With Depth-Guided Scoring**:

| Benchmark Dataset | Method / Architecture | Without Depth AUC (%) | With Depth AUC (%) |
| :--- | :--- | :---: | :---: |
| **UCSD Ped2** | Ours ( |**95.16%** |  |
| **Avenue** | Ours  | 84.84% | **85.81%** |
| **SUT Anomaly** | Ours| 69.61% | **74.39%** |

---

## Environment & Setup

### 1. Installation
```bash
git clone https://github.com/your-username/CMP_3CNN_VAD.git
cd CMP_3CNN_VAD
pip install -r requirements.txt
```

### 2. Depth Estimation Setup (SC-Depth Integration)
Evaluation with depth-guided scoring relies on monocular depth estimation from [SC-Depth (V3)](https://github.com/JiawangBian/sc_depth_pl). Please complete the following two steps before evaluation:

1. **Clone the SC-Depth Repository:**
   Clone the `sc_depth_pl` repository directly into the root folder as `sc_depth_pl_master`:
   ```bash
   git clone https://github.com/JiawangBian/sc_depth_pl.git sc_depth_pl_master
   ```

2. **Download Pretrained Depth Model Weights:**
   - Download the pretrained DDAD checkpoint (`epoch=99-val_loss=0.1438.ckpt`) from the [SC-Depth GitHub Repository](https://github.com/JiawangBian/sc_depth_pl).
   - Place the downloaded checkpoint file into the target relative path:
     ```text
     sc_depth_pl_master/ckpts/ddad_scv3/epoch=99-val_loss=0.1438.ckpt
     ```

---

## Directory & Data Structure

Arrange your training and testing frame folders under your root dataset path as follows:

```text
data/
├── Avenue/
│   └── Avenue_Dataset/
│       ├── Train/
│       └── Test/
├── UCSDped2/
│   ├── Train/
│   └── Test/
└── SUT_data/
    ├── Train/
    └── Test/ 
```

---

## How to Run (Training & Evaluation)

All routines are executed via `main.py` using command-line flags matching the repository parameters.

### 1. Training
The framework trains on normal video clips using a two-stage scheme: initial autoencoder reconstruction optimization followed by latent channel-wise compactness regularization starting after 1/4 of total iterations.

To start a training run:
```bash
python main.py --gpu 0 --dataset_name avenue --folder_path "Avenue/Avenue_Dataset/Train" --compact 1 --pretrain_tag 1
```

**Key Training Command Arguments:**
* `--gpu`: GPU index selection (default: `0`).
* `--dataset_name`: Name of target dataset (`avenue`, `ped2`, `SUT`).
* `--folder_path`: Relative path to training frames directory.
* `--compact`: Enable channel-wise feature compactness loss (`1` or `0`, default: `1`).
* `--log_path`: Output directory for checkpoints and logs (default: `./log`).

---

### 2. Evaluation
To evaluate a trained model and compute frame-level AUC-ROC scores across test video sequences (uses the loaded SC-Depth module for depth refinement):

```bash
python main.py --gpu 0 --dataset_name avenue --Test_Folder "Avenue/Avenue_Dataset/Test" --eval 1
```

**Key Evaluation Command Arguments:**
* `--eval`: Set to `1` to run model evaluation.
* `--Test_Folder`: Relative path to evaluation frames directory.
* `--dataset_path`: Root directory for dataset (default: current directory).

---

<!--
## Citation

```bibtex
@article{namin2020depth,
  title={Depth-guided video anomaly detection using compact representation of spatio-temporal features},
  author={Namin, Habib Ebadi and Kharghanian, Reza and Ahmadyfard, Alireza},
  journal={Journal of LaTeX Class Files},
  volume={18},
  number={9},
  year={2020}
}
```
-->
