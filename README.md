---
## 📄 Publication and Code Status

**The methods and experimental results in this repository are directly related to the academic paper titled "Improving Multi-Object Tracking Robustness with Adaptive Interpolation and Weak Cue Integration," which has been submitted to The Visual Computer journal.**

## Introduction

This repository contains the official implementation of thesis **"Improving Multi-Object Tracking Robustness with Adaptive Interpolation and Weak Cue Integration"**.

Multi-object tracking (MOT) in complex dynamic scenarios—such as **DanceTrack**—faces significant challenges, including **non-rigid motion**, **weak spatial cues**, and **appearance homogenization** (unified appearance). Built upon the strong baseline of [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), this project introduces a robust tracking framework that integrates a **Motion Optimization Module** and a **Weak-Cue Fusion Association Strategy**.

Our approach significantly improves robustness against non-linear motion and occlusion, achieving state-of-the-art performance on **DanceTrack**, **MOT17**, and **MOT20** benchmarks among non-transformer methods.



-----

## 💡 Technical Deep Dive: Algorithm Implementation

We propose innovations in two key aspects: **State Estimation (Motion Modeling)** and **Data Association**.


### 1\. Motion Model Optimization (Kalman Filter Modifications)

  * **Dynamic-Weight Linear Interpolation (DWLI):** This interpolation method is applied during **track recovery** (after a short occlusion or miss). Unlike standard linear interpolation, DWLI assigns a **dynamic, unequal weight** (based on track confidence and age) separately to the position components ($x, y$) and the scale components ($w, h$) during trajectory smoothing.

### 2\. Weak Cue Association Strategy (Cost Matrix Fusion)

These modules are integrated into the association stage to increase robustness when primary cues (IoU, appearance) are ambiguous:

  * **Height-Modulated IoU (HMIoU):** This method modulates the standard IoU cost based on the **relative stability of the target's height**. Height change is generally slower than width change in non-rigid motion. The HMIoU cost function incorporates this height factor to prioritize stable vertical scale matches.
  * **Confidence State Tracking (CST):** The track's **detection confidence** is modeled as a state input. This allows the association cost to factor in the temporal stability of the detector's score, assisting in tracking through brief periods of weak detection or clutter.
  * **Pseudo-Depth Association (PDA):** The cost matrix integrates a feature derived from the bounding box's **vertical position** ($y$-coordinate). This "pseudo-depth" leverages the ground plane prior ("objects lower on the screen are closer") to break ties in crowded, low-perspective scenes.

-----

## Benchmark Results

### DanceTrack Test Set

> Evaluating performance on diverse motion and uniform appearance.

| Method | HOTA | IDF1 | AssA | MOTA |
|:---|:---:|:---:|:---:|:---:|
| OC-SORT | 55.1 | 54.2 | 38.0 | 89.4 |
| Deep OC-SORT | 61.3 | 61.5 | 45.8 | 92.3 |
| **Ours** | **63.2** | **65.3** | **49.0** | **91.9** |

### MOT17 Test Set

> General pedestrian tracking.

| Method | HOTA | IDF1 | AssA | MOTA |
|:---|:---:|:---:|:---:|:---:|
| ByteTrack | 63.1 | 77.3 | 62.0 | 80.3 |
| Deep OC-SORT | 64.9 | 80.6 | 65.9 | 79.4 |
| **Ours** | **65.1** | **80.6** | **66.3** | **79.4** |

### MOT20 Test Set

> Extreme density and occlusion.

| Method | HOTA | IDF1 | AssA | MOTA |
|:---|:---:|:---:|:---:|:---:|
| ByteTrack | 63.1 | 77.3 | 62.0 | 80.3 |
| Deep OC-SORT | 64.9 | 80.6 | 65.9 | 79.4 |
| **Ours** | **65.0** | **80.8** | **67.5** | **76.1** |





## 💻 Dependencies and Requirements

To ensure full reproducibility, please install the following environment components:

| Component | Recommended Version | Note |
|:---|:---|:---|
| **Python** | 3.8+ | |
| **PyTorch** | 1.10.1 | Tested with CUDA 11.3 |
| **CUDA** | 11.3 / 11.8 | Requires GPU for running YOLOX detector |
| **OS** | Linux (e.g., Ubuntu 20.04) | |

## Installation

This codebase is based on PyTorch.

1.  **Clone the repository**

    ```bash
    git clone https://github.com/fei528/sy-dwli-wcf.git
    cd sy-dwli-wcf
    ```

2.  **Environment Setup**

    ```bash
    conda create -n mot_tracker python=3.8
    conda activate mot_tracker
    pip install -r requirements.txt
    ```

3.  **Setup**

    ```bash
    python3 setup.py develop
    ```

## Data Preparation

Download the datasets from their official websites and organize them as follows:

  * **MOT17 / MOT20:** [MOTChallenge](https://motchallenge.net/)
  * **DanceTrack:** [DanceTrack](https://dancetrack.github.io/)

<!-- end list -->

```
datasets
├── mot
│   ├── train
│   │   ├── MOT17-02-DPM
│   │   ├── ...
│   ├── test
│   │   ├── MOT17-01-DPM
│   │   ├── ...
├── dancetrack
│   ├── train
│   ├── val
│   ├── test
```

## Inference & Evaluation

To run the tracker on specific datasets:

**DanceTrack:**

```bash
python3 tools/track.py --benchmark dancetrack --eval --exp_name dance_exp --fp16 --fuse
```

**MOT17:**

```bash
python3 tools/track.py --benchmark mot17 --eval --exp_name mot17_exp --fp16 --fuse
```

**MOT20:**

```bash
python3 tools/track.py --benchmark mot20 --eval --exp_name mot20_exp --fp16 --fuse
```


### 📢 Citation Request

To ensure the transparency and reproducibility of this research, we urge any reader using this code or any work derived from it to **cite this forthcoming paper in any work using this code (even if it is currently in submission status)** (citation information will be updated after the visual computer officially publishes it). Your citations will greatly support our academic research and future development.



## Acknowledgement

This project is deeply indebted to the following open-source works:

  * [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)
  * [OC-SORT](https://github.com/noahcao/OC_SORT)
  * [ByteTrack](https://github.com/ifzhang/ByteTrack)
  * [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
