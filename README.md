# Agent Memory Decay: Theoretical Framework and Empirical Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://zenodo.org/doi/PLACEHOLDER)

Companion code and data for the paper:

> **Agent Memory Decay: Theoretical Framework and Empirical Benchmark for Continuous-Time vs Discrete Models in LLM Agent Memory Systems**
>
> Submitted to *PeerJ Computer Science* (2026)

## Overview

This repository provides:
- **8 decay model implementations** (LTC, CfC, Neural ODE, CCT-LNN, GRU, GRU+Time, Parametric ExpDecay, Transformer)
- **3 complementary evaluation paradigms** (regression, focal weighted regression, per-timestep ranking)
- **Production OS agent memory system** with LTC gate deployed on Kylin Linux V10
- **58+ experiments** across 8 categories (Exp23-80)
- **All experimental results** from the paper

## Repository Structure

```
.
├── models/
│   └── v21_models.py              # 8 decay model implementations (PyTorch)
├── src/
│   └── gate/
│       ├── lnn_model.py           # LTC gate architecture
│       ├── memory_gate.py         # Gate integration with memory retrieval
│       └── lnn_trainer.py         # Gate training and evaluation
├── os_agent_system/
│   ├── gate/                      # LTC gate (production version)
│   ├── memory/                    # Memory management (forgetting curve, tiers, manager)
│   ├── search/                    # BM25 + hybrid search engine
│   ├── knowledge/                 # Knowledge engine (24-type conflict detection)
│   ├── extraction/                # Zero-LLM pipeline
│   └── security/                  # Sensitive filter (PII/security)
├── experiments/
│   ├── v21_c0_fix.py              # C0 fix: Exp-1 (regression) + Exp-2 (balanced) + Exp-3 (ranking)
│   ├── v21_c0_exp3_only.py        # Exp-3 standalone (memory-efficient)
│   ├── v21_c0_fix_efficient.py    # Lightweight version
│   ├── exp_personachat.py         # PersonaChat experiments
│   ├── exp_lpt.py                 # LPT experiments
│   ├── run_stress_tests.py        # Stress tests (Exp48-60)
│   └── _exp*_*.py                 # Additional experiments
├── data/
│   ├── personachat_enhanced.json  # PersonaChat dataset (100 users, 8 categories)
│   └── lpt_200users.json          # LPT dataset (200 users, 5 categories)
├── results/
│   ├── v21_c0_fixed_results.json  # Complete 3-experiment results (Exp-1/2/3)
│   ├── v21_c0_exp3_results.json   # Exp-3 ranking results
│   ├── personachat_results.json   # PersonaChat baseline results
│   ├── lpt_results.json           # LPT baseline results
│   ├── results_exp38_47.json      # Module coverage experiments
│   ├── results_exp48_55.json      # Adversarial experiments
│   ├── results_exp56_60.json      # Targeted experiments
│   ├── results_stress.json        # Stress test results
│   └── results_m1_ablation.json   # Gate ablation results
├── figures/                       # Paper figures (12 PNG files)
├── supplementary/
│   ├── v22_PeerJCS.docx           # Paper manuscript (PeerJ CS format)
│   ├── v22_peerj_paragraphs.txt   # Paper text (for programmatic access)
│   ├── v22_self_review.md         # Self-review report
│   └── PeerJCS_Review_v21p.md    # Prior review (KBS)
├── paper/
│   └── v21p_PeerJCS_Paper.docx    # Earlier version
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick Start

### Requirements
- Python 3.11+
- PyTorch 2.0+ (CUDA recommended)
- NumPy, SciPy

### Install
```bash
git clone https://github.com/PLACEHOLDER/agent-memory-decay-benchmark.git
cd agent-memory-decay-benchmark
pip install -r requirements.txt
```

### Run Experiments
```bash
cd experiments

# Full C0 fix (Exp-1 + Exp-2 + Exp-3, ~30 min on GPU)
python v21_c0_fix.py

# Exp-3 only (memory-efficient, ~5 min on CPU)
python v21_c0_exp3_only.py
```

### Load Pre-computed Results
```python
import json

with open("results/v21_c0_fixed_results.json") as f:
    results = json.load(f)

# Exp-3 ranking results
print("=== Exp-3: Per-Timestep Ranking Quality ===")
for model, metrics in results["exp3_ranking"]["personaChat"].items():
    print(f"  {model}: NDCG@5={metrics['ndcg5']:.3f}, MRR={metrics['mrr']:.3f}")
```

## Key Results

### Exp-3: Per-Timestep Ranking Quality (NDCG@5, random=0.500)

| Model | NDCG@5 | MRR | Kendall τ |
|-------|--------|-----|-----------|
| Neural ODE | **0.616** | 0.553 | 0.195 |
| GRU | 0.604 | 0.543 | 0.184 |
| GRU+Time | 0.603 | **0.559** | **0.196** |
| CfC | 0.586 | 0.509 | 0.125 |
| LTC | 0.585 | 0.415 | 0.083 |
| Transformer | 0.546 | 0.352 | -0.016 |
| CCT-LNN | 0.536 | 0.355 | 0.002 |
| Param. ExpDecay | 0.533 | 0.324 | -0.048 |

### Production Deployment (Kylin Linux V10)

| Metric | Threshold | Result |
|--------|-----------|--------|
| Preference Accuracy | >=85% | 86.6% |
| Knowledge Retrieval R@5 | >=85% | 93.1% |
| Retrieval Latency P95 | <=500ms | 100.4ms |
| Conflict Detection | >=88% | 93.3% |
| Memory Footprint | - | 16.8 MB |

## Datasets

### PersonaChat (included)
- 100 users, 8 categories, avg 33 entries/user
- Source: [PersonaChat](https://arxiv.org/abs/1801.07243)
- Pre-processed with 64-dimensional embeddings
- File: `data/personachat_enhanced.json` (6.1 MB)

### LPT (included)
- 200 users, 5 categories, avg 189 entries/user, 63-day span
- File: `data/lpt_200users.json` (24.2 MB)

## Citation

```bibtex
@article{PLACEHOLDER,
  title={Agent Memory Decay: Theoretical Framework and Empirical Benchmark 
         for Continuous-Time vs Discrete Models in LLM Agent Memory Systems},
  author={PLACEHOLDER},
  journal={PeerJ Computer Science},
  year={2026},
  doi={10.5281/zenodo.PLACEHOLDER}
}
```

## License

MIT License - see [LICENSE](LICENSE)
