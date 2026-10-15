# LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

An end-to-end implementation of **LEADRE**, an industrial generative display ad retrieval system developed by **Tencent Inc.** and **Renmin University of China** (PVLDB 2025 / arXiv:2411.13789v3), deployed on WeChat Channels and Moments serving tens of billions of requests daily.

---

## 🏛️ System Architecture

```text
                               +-------------------------------------------------------------+
                               |                 LEADRE System Overview                      |
                               +-------------------------------------------------------------+
                                                             |
              +------------------------------+---------------+-----------------------------+
              |                              |                                             |
              v                              v                                             v
  [1. Ads Indexing (RQ-VAE)]     [2. Intent-Aware Prompting]                  [3. Hybrid Serving Architecture]
  • Dense Text Encoding (f_PLM)  • Learnable Soft Token (<SoftPrompt>)        • Online Cache (<10ms SLA)
  • 4-Layer Residual Quantizer   • User Demographics Profile                  • Nearline LLM Trie Beam Search
  • Disambiguation Token (e_*)   • Top Category Interaction Summary           • Adaptive ARPU Prioritization
  • Discrete S-IDs (<a,b,c,d,e>) • 90-Day Merged Ad & Content Behaviors       • Downstream Ranking Features
  • Prefix Trie-Tree Index       • Sequence Augmentations (Prefix Slicing)      (+1.43% GMV gain)
```

---

## 📁 Repository Structure

```text
LEADRE/
├── configs/
│   ├── rqvae_config.yaml            # RQ-VAE codebook size, layers, and dimensions
│   ├── curriculum_training.yaml     # 3-Stage learning rates, epochs, and prompt max len
│   ├── dpo_config.yaml              # DPO temperature beta and revenue margin settings
│   └── serving_config.yaml          # Online SLA limits, cache TTL, and nearline beam size
├── leadre/
│   ├── data/
│   │   ├── schemas.py               # Pydantic schemas for Ads, Profiles, and Event Sequences
│   │   ├── prompt_builder.py        # Multi-component prompt constructor & augmentations
│   │   └── dataset_generator.py     # Display ads synthesizer & 4-stage dataset creator
│   ├── models/
│   │   ├── feature_encoder.py       # PLM embedding wrapper (f_PLM: Eq. 1)
│   │   ├── rqvae.py                 # Residual-Quantized VAE (Eq. 2-6) with S-ID assignment
│   │   ├── trie.py                  # Algorithm 1: Prefix Trie-Tree construction & lookup
│   │   └── constrained_logits.py    # Algorithm 2: Constrained Beam Search & LogitsProcessor
│   ├── training/
│   │   ├── train_rqvae.py           # RQ-VAE trainer with codebook collapse monitoring
│   │   ├── curriculum_trainer.py    # 3-Stage sequential training (EX -> IM -> Main)
│   │   └── dpo_trainer.py           # Direct Preference Optimization (DPO) with ECPM pairs
│   ├── serving/
│   │   ├── online_cache.py          # Sub-10ms user candidate cache & Mixer service
│   │   ├── nearline_service.py      # Nearline inference worker with ARPU adaptive scheduling
│   │   └── ranking_features.py      # Downstream user interest & item match features
│   └── evaluation/
│       └── metrics.py               # HR@K, NDCG@K, Diversity Score, and Dice Similarity
├── LLM based RS.pdf                 # Original research paper
├── requirements.txt                 # Python dependencies
├── run_pipeline.py                  # End-to-end runnable verification pipeline
└── README.md                        # Documentation
```

---

## 🚀 Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete End-to-End Pipeline
```bash
python run_pipeline.py
```

This single command automatically executes and validates all 6 phases:
1. **Catalog & User Simulation:** Generates ad items and 90-day chronological ad & content sequences.
2. **RQ-VAE Ads Indexing:** Fits vector quantizers ($M=4$ codebooks, $K=1024$), checks codebook usage, assigns discrete S-IDs, and builds the Prefix Trie-Tree (Algorithm 1).
3. **Curriculum Dataset Preparation:** Formulates samples for Explicit Alignment (EX), Implicit Alignment (IM), Main Generation, and DPO pairs.
4. **Curriculum Training & DPO:** Runs the optimal sequential curriculum (EX $\to$ IM $\to$ Main) and executes DPO policy alignment for ECPM revenue gains.
5. **Hybrid Serving Simulation:** Nearline worker executes Algorithm 2 (Trie-Constrained Beam Search) and populates the cache; online mixer retrieves candidates in $<0.01\text{ ms}$ (passing the $<10\text{ ms}$ SLA).
6. **Downstream Feature Extraction & Offline Evaluation:** Computes user interest vectors, item match features, and benchmarks `HR@1`, `HR@4`, `HR@8`, `NDCG@1`, `NDCG@4`, `NDCG@8`, Abundance, and Concentration.

---

## 🔬 Core Methodological Innovations

### 1. Residual-Quantized Variational AutoEncoder (RQ-VAE)
Discrete S-IDs solve the semantic gap between free-form LLM text and fixed advertising inventories:
- Codebooks: $M=4$ levels with $K=1024$ vectors of dimension $d_{RQ}=8$.
- Disambiguation: S-IDs sharing identical prefixes receive a 5th collision disambiguation token $e_*$.
- Output format: `<a_808, b_380, c_413, d_25, e_0>`.

### 2. Sequential Curriculum Alignment (EX $\to$ IM $\to$ Main)
The paper proves that multi-task training order directly determines performance:
- **Stage 1 (EX):** Ad Text Description $\to$ S-ID.
- **Stage 2 (IM):** User Sequence with Ad Text $\to$ Next Ad S-ID.
- **Stage 3 (Main):** Full Intent Prompt with S-IDs $\to$ Next Ad S-ID.

### 3. Business Objective Alignment (DPO)
Relevance alone is insufficient in advertising; ads have different ECPM values. Direct Preference Optimization constructs preference pairs $\langle u, a_h, a_l \rangle$ with $\text{ECPM}(a_h) > \text{ECPM}(a_l)$:
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(u, a_h, a_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(a_h|u)}{\pi_{ref}(a_h|u)} - \beta \log \frac{\pi_\theta(a_l|u)}{\pi_{ref}(a_l|u)} \right) \right]$$

### 4. Nearline Pre-computation with ARPU Priority
Online display ad SLA requires $<10\text{ ms}$ latency. Generative beam search is decoupled into an asynchronous nearline worker:
- Users are grouped into 25 ARPU buckets.
- GPU inference is prioritized for top-value tiers.
- Generated candidates are cached in Redis for instantaneous online retrieval.

### 5. Downstream Ranking Feature Enrichment
Recalled ads are converted into:
- **User-Side Feature:** Pooled interest embedding of retrieved ads.
- **Item-Side Feature:** Cosine similarity and maximum match score with candidate ads.
- Yielded an additional **+1.43% GMV gain** in live WeChat Channels A/B testing.

---

## 📊 Citation
```bibtex
@article{li2025leadre,
  title={LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System},
  author={Li, Fengxin and Li, Yi and Liu, Yue and Zhou, Chao and Wang, Yuan and Deng, Xiaoxiang and Xue, Wei and Liu, Dapeng and Xiao, Lei and Gu, Haijie and Jiang, Jie and Liu, Hongyan and Qin, Biao and He, Jun},
  journal={Proceedings of the VLDB Endowment},
  volume={18},
  number={12},
  year={2025}
}
```
