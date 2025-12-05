\<div align="center"\>
\<p\>
\<a href="[https://github.com/fei528/sy-dwli-wcf](https://github.com/fei528/sy-dwli-wcf)"\>\<img src="[https://img.shields.io/badge/Github-Code-blue](https://www.google.com/search?q=https://img.shields.io/badge/Github-Code-blue)" alt="Github Code"\>\</a\>
\<a href="\#"\>\<img src="[https://img.shields.io/badge/Paper-Thesis-green](https://www.google.com/search?q=https://img.shields.io/badge/Paper-Thesis-green)" alt="Paper"\>\</a\>
\<a href="\#"\>\<img src="[https://img.shields.io/badge/Task-MOT-red](https://www.google.com/search?q=https://img.shields.io/badge/Task-MOT-red)" alt="Task"\>\</a\>
\</p\>
\</div\>

## Introduction

This repository contains the official implementation of thesis **"Improving Multi-Object Tracking Robustness with Adaptive Interpolation and Weak Cue Integration"**.

Multi-object tracking (MOT) in complex dynamic scenarios—such as **DanceTrack**—faces significant challenges, including **non-rigid motion**, **weak spatial cues**, and **appearance homogenization** (unified appearance). Built upon the strong baseline of [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), this project introduces a robust tracking framework that integrates a **Motion Optimization Module** and a **Weak-Cue Fusion Association Strategy**.

Our approach significantly improves robustness against non-linear motion and occlusion, achieving state-of-the-art performance on **DanceTrack**, **MOT17**, and **MOT20** benchmarks among non-transformer methods.

## Key Features

We propose innovations in two key aspects: **State Estimation (Motion Modeling)** and **Data Association**.

### 1\. Motion Model Optimization (Chapter 3)

  
  * [cite_start]**Dynamic-Weight Linear Interpolation (DWLI):** A novel interpolation method that assigns adaptive weights to position and scale dimensions independently, effectively repairing trajectory breaks caused by non-linear deformation[cite: 263].

### 2\. Weak Cue Association Strategy (Chapter 4)

  * [cite_start]**Height-Modulated IoU (HMIoU):** Leveraging the observation that object **height** is more stable than width during occlusion, we introduce HMIoU to enhance association robustness in crowded scenes[cite: 355].
  * [cite_start]**Confidence State Tracking (CST):** We incorporate detection confidence and its rate of change into the state vector, utilizing the temporal continuity of confidence to assist tracking during occlusion or detector degradation[cite: 367].
  * [cite_start]**Pseudo-Depth Association (PDA):** Utilizing the geometric prior of monocular perspective ("near is low, far is high"), we construct a pseudo-depth feature to resolve depth ambiguities for targets with similar appearances[cite: 387].

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



## Acknowledgement

This project is deeply indebted to the following open-source works:

  * [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)
  * [OC-SORT](https://github.com/noahcao/OC_SORT)
  * [ByteTrack](https://github.com/ifzhang/ByteTrack)
  * [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
