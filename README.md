# ☀️ PBCAT-M: Physics-Based Convolutional Attention Transformer with Mamba

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)

> *"The only model that tells you HOW CONFIDENT it is alongside every prediction."*

**PBCAT-M** is a state-of-the-art, dual-horizon deep learning architecture built specifically for the **ISRO Space Weather Monitoring Center**. It predicts solar flares (C, M, and X-class) using raw X-ray data directly from the **Aditya-L1** mission (SoLEXS and HEL1OS payloads). 

Unlike standard black-box AI, PBCAT-M incorporates **Mamba State Space Models**, **Physics-informed Asymmetric Cross-Attention**, and **Mahalanobis Out-Of-Distribution (OOD) detection** to provide high-speed, highly reliable space weather forecasts every 30 seconds.

---

## 🚀 Key Features

* **Real-time Processing**: Ingests, cleans, and infers 6-hour windows in milliseconds.
* **Dual-Horizon Forecasting**: Predicts flares happening *now* (Nowcasting) and *24-hours ahead* (Forecasting) simultaneously.
* **Mamba SSM Backbone**: Replaces traditional slow Transformers with highly efficient `mamba-ssm` sequence modeling for processing long time-series data without memory bottlenecks.
* **Physics-Informed Fusion**: Treats SoLEXS (soft X-ray) as the baseline and HEL1OS (hard X-ray) as the high-energy "spark" using asymmetric cross-attention.
* **Extreme Event Flagging**: Utilizes Isolation Forests and Mahalanobis distance to flag unprecedented sensor anomalies, preventing false "X-class" alarms during hardware glitches.

---

## 🏗️ Architecture (The 7 Stages)

The PBCAT-M network is structured into 7 modular stages:

1. **Stage 1: Raw Data Ingestion & Preprocessing** ✅
   * Extracts `.fits` data, applies Good Time Intervals (GTI), removes spikes via rolling medians, and embeds 6-hour cadences into PyTorch tensors using `PatchEmbedding`.
2. **Stage 2: CNN Encoder** 🚧 *(In Progress)*
   * Per-channel local feature extraction using 1D Convolutions.
3. **Stage 3: Asymmetric Cross-Attention (SoLEXS → HEL1OS)** 
   * Fuses soft and hard X-ray features using PyTorch Multi-Head Attention.
4. **Stage 4: Mamba SSM Backbone**
   * Long-sequence temporal modeling using the highly optimized Mamba architecture.
5. **Stage 5: LoRA Fine-Tuning Module**
   * Domain adaptation for Aditya-L1 data using HuggingFace PEFT.
6. **Stage 6: OOD Detection & Flagging**
   * Identifies statistical anomalies and missing sensor data gaps to output confidence bounds.
7. **Stage 7: Multi-Instance Dual Output Heads**
   * Final dense layers split to predict both "Now" and "24h Horizon" flare classifications.

---

## ⚙️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch 2.x | Core framework |
| **Sequence Backbone** | `mamba-ssm` | Fast, long-sequence modeling |
| **CNN Encoder** | PyTorch `nn.Conv1d` | Local feature extraction |
| **Attention** | PyTorch `nn.MultiheadAttention`| SoLEXS → HEL1OS fusion |
| **OOD Detection** | Scikit-Learn | Extreme event anomaly flagging |
| **Fine-Tuning** | HuggingFace PEFT (LoRA) | Domain adaptation |
| **Data Processing** | Pandas, NumPy, Astropy | Preprocessing & FITS parsing |

---

## 💻 Running the Pipeline

Currently, you can verify the Stage 1 End-to-End ingestion pipeline using the test script:

```bash
pip install -r requirements.txt
python test_pipeline.py
```

This will run a 6-hour simulated window through the Data Preprocessor and PyTorch PatchEmbedding layer, verifying the expected `[1, 720, 256]` tensor output.

---

## 📝 Future Work
*See `brain.md` for deferred tasks, including NOAA/GOES label integration and the custom PyTorch Training Loop setup.*
