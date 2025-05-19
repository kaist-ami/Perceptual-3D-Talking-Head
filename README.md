# Perceptually Accurate 3D Talking Head Generation: New Definitions, Speech-Mesh Representation, and Evaluation Metrics 
<h3>CVPR 2025 <mark>Highlight</mark></h3>


### [Project Page](https://perceptual-3d-talking-head.github.io/) | [Paper](https://arxiv.org/pdf/2503.20308)

![Image](https://github.com/user-attachments/assets/90a114a5-5bc0-49dc-bb3b-b069784e4328)

<div align="center">
We define three criteria to assess perceptual alignment between speech and lip movements of 3D talking heads: <br>
Temporal Synchronization, Lip Readability, and Expressiveness.
</div>
<br>

This repository includes **speech-mesh synchronized representation** and their usage as **a perceptual loss**. 
We also provide **the evaluation codes for three metrics**—MTM, PLRS, and SLCC—to assess how well the generated 3D talking heads align with the three criteria.

# Evaluation

This directory provides three evaluation pipelines:

1. **Mean Temporal Misalignment (MTM)** – temporal discrepancy between speech and corresponding lip movements.
2. **Perceptual Lip Readability Score (PLRS)** -  perceptual alignment between lip movements and speech.
3. **Speech-Lip Intensity Correlation Coefficient (SLCC)** – expressiveness correlation between lip movements and speech

---

## 🕒 Mean Temporal Misalignment (MTM)

This script computes the **Mean Temporal Misalignment** between ground-truth and predicted vertex sequences.  
Example `.npy` files (ground-truth / FaceFormer predictions) are included. 

Note that the metric also supports **one-to-many comparisons**—e.g. a single ground-truth sequence vs. multiple predictions conditioned on speaker identity.

### Run

```bash
cd evaluation
python evaluate_MTM.py
```

### Output

* A **CSV** file per clip containing  
  * **Mean Δt (frames)** – average temporal offset  
  * **# matching points** – matched vertex pairs  
  * **Δt per point** – frame-wise misalignment  
* A **PNG** visualization for each clip.

### Convert to ms

If your dataset is **25 FPS**:

```
Δt (ms) = Δt (frames) × 40 ms
```

---

## 📈 Speech-Lip Intensity Correlation Coefficient (SLCC)

This pipeline correlates **speech intensity** (audio RMS) with **lip-motion intensity** (vertex displacement) to quantify expressiveness.

### ①  Download MEAD

1. Grab MEAD from **[Google Drive](https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j)**.  
2. Place it here:

```
evaluation/MEAD
```

3. Directory must look like:

```
evaluation/
└── MEAD
    ├── M030
    │   ├── images
    │   └── video
    │       ├── front
    │       ├── down
    │       ├── left_30
    │       ├── left_60
    │       ├── right_30
    │       ├── right_60
    │       └── ...
    │           └── angry
    │               ├── level_1
    │               │   ├── 001.mp4
    │               │   └── ...
    │               ├── level_2
    │               └── level_3
    ├── M031
    └── ...
```

### ②  Extract Speech Intensity (SI)

```bash
cd evaluation
python extract_rms.py
```

This writes an **RMS CSV** for every video clip.

### ③  Extract Lip Intensity (LI)

1. Put your predicted vertex files in:

```
evaluation/data_SLCC/
```

2. File-name format **(required)**:

```
{id}_{emotion}_{level}_{clip}_condition_{condition_id}.npy
```

*Example*

```
M035_angry_level_2_001_condition_FaceTalk_170725_00137_TA.npy
```

3. Run:

```bash
python extract_lip_intensity.py
```

This produces a **lip-displacement CSV** for each clip.

### ④  SLCC Evaluation

```bash
python extract_lip_intensity.py   # second pass computes SLCC
```

#### Results

* **Overall SLCC**  
* **SLCC per expression level** (`level_1`, `level_2`, `level_3`)

Plots and summary tables are saved to `SLCC_results/`.

---


## 📚 Citation
If you found this code useful, please consider citing our paper.

```
@article{chae2025perceptually,
  title={Perceptually Accurate 3D Talking Head Generation: New Definitions, Speech-Mesh Representation, and Evaluation Metrics},
  author={Chae-Yeon, Lee and Hyun-Bin, Oh and EunGi, Han and Sung-Bin, Kim and Nam, Suekyeong and Oh, Tae-Hyun},
  journal={arXiv preprint arXiv:2503.20308},
  year={2025}
}
```
