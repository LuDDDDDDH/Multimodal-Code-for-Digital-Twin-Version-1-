# Hepato-Multimodal-Transformer (HMT) System

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-green)](https://github.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Transformer-blue)](https://github.com/)

## 📖 Introduction (项目简介)

**Hepato-Multimodal-Transformer (HMT)** is an advanced AI framework for the precision prognosis of Hepatocellular Carcinoma (HCC). It bridges the gap between high-dimensional 3D medical imaging and heterogeneous clinical omics data.

This repository hosts the evolution of our research from a robust **Decoupled Baseline (v1.0)** to a next-generation **"Avatar" Cross-Attention Architecture (v2.0)**.

---

## 🌟 Version Highlights (版本亮点)

### 🧬 v2.0: The "Avatar" Architecture (Current Dev)
> **Branch:** `dev-v2-avatar` | **Core:** `avatar_model.py`

*   **Dual-Stream Engine:** Specialized `Spatial Stream` for CT imaging and `Avatar Stream` for omics data.
*   **Semantically-Guided Attention:** Uses clinical/omics tokens as "Queries" to retrieve relevant visual features from the 3D CT "Keys/Values".
*   **Resolution Agnostic:** Handles variable input sizes naturally via attention mechanisms.
*   **Mock-Pro Demo:** Includes a professional-grade demo script (`demo_avatar_pro.py`) that simulates the full system pipeline without external dependencies.

### 🏛️ v1.0: Decoupled Baseline
> **Branch:** `main` | **Core:** `model.py` + `train_heavy.py`

*   **Heavy/Light Strategy:** A resource-efficient pipeline that separates 3D feature extraction (Swin UNETR) from multimodal fusion.
*   **Missing Token Learning:** Innovative handling of missing clinical modalities using learnable embeddings.
*   **Multi-Task Heads:** Simultaneous prediction of Efficacy (RECIST), Toxicity, and Survival (Cox).

---

## 🚀 Quick Start (快速开始)

We provide demonstration scripts for both versions to verify the architecture without needing real patient data.

### To Run v2.0 "Avatar" Demo (Recommended)
Switch to the dev branch and run the professional demo:
```bash
git checkout dev-v2-avatar
python demo_avatar_pro.py
Expected Output: System initialization logs, mock data generation, and risk score prediction with attention weights.
To Run v1.0 Baseline Demo
Switch to the main branch and run the legacy demo:
code
Bash
git checkout main
python demo_v1_legacy.py
🏗 System Architecture (系统架构图)
v2.0 Cross-Attention Flow
code
Mermaid
graph TD
    subgraph Omics Stream
    A[Clinical/Blood/Urine] --> B(Projectors)
    B --> C[Omics Queries]
    end
    
    subgraph Spatial Stream
    D[3D CT Image] --> E[Swin UNETR Backbone]
    E --> F[Visual Keys/Values]
    end
    
    C & F --> G{Cross-Attention Engine}
    G --> H[Visual-Enhanced Omics Features]
    H --> I[Risk Prediction Heads]
📂 Repository Structure
code
Text
.
├── avatar_model.py           # [v2.0] New Cross-Attention Architecture
├── demo_avatar_pro.py        # [v2.0] System Simulation Script
├── demo_v1_legacy.py         # [v1.0] Baseline Verification Script
├── config.py                 # Configuration Hub
├── model.py                  # [v1.0] Original Transformer Model
├── train_heavy/light.py      # Decoupled Training Scripts
├── run_full_experiment.sh    # Automation Bash Script
└── README.md                 # Documentation
📧 Contact
Donghai Lu
Project Lead & Algorithm Engineer
Affiliation: Shandong University / Qilu Hospital
Focus: AI for Medical Imaging, Multimodal Fusion