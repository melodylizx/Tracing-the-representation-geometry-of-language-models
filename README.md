# Tracing the Representation Geometry of Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.23024-b31b1b.svg)](https://arxiv.org/abs/2509.23024)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/119054)

Official code for our NeurIPS 2025 paper on analyzing LLM training through geometric phases.

## Overview

We use spectral methods (RankMe and α-ReQ) to track how LLM representations evolve during training. We find three consistent phases:

1. **Warmup**: Rapid representational collapse
2. **Entropy-seeking**: Expansion of dimensionality, peak memorization
3. **Compression-seeking**: Anisotropic consolidation, improved generalization

## Installation

```bash
pip install torch transformers datasets numpy matplotlib seaborn scikit-learn tqdm jsonargparse
```

## Quick Start

### Compute metrics for Pythia models

```bash
python rankme_alpha_scripts/compute_alpha_pythia.py --model_name pythia-410m
```

### Compute metrics for OLMo models

```bash
python rankme_alpha_scripts/compute_alpha_rankme_olmo2.py \
    --model_name OLMo-1B \
    --batch_size 32
```

### Visualize results

```bash
python analysis/plot_alpha_traj.py --xvar steps
```

## Repository Structure

```
├── analysis/                       # Visualization scripts
│   └── plot_alpha_traj.py
├── memorization_scripts/           # N-gram and LLM likelihood analysis
│   ├── compute_infgrams.py
│   └── compute_llm_likelihood.py
├── rankme_alpha_scripts/          # Compute geometric metrics
│   ├── compute_alpha_pythia.py
│   ├── compute_rankme_pythia.py
│   └── compute_alpha_rankme_olmo2.py
└── utils/
    ├── powerlaw.py                # Eigenspectrum and metric utilities
    └── llm_prob.py                # Token probability analysis
```

## Key Metrics

**RankMe (Effective Rank)**
- Measures effective dimensionality of representations
- Higher = more isotropic (entropy-seeking phase)
- Lower = more collapsed (compression-seeking phase)

**α-ReQ (Power-law Exponent)**
- Measures eigenspectrum decay rate
- Lower α = more uniform (entropy-seeking)
- Higher α = more concentrated (compression-seeking)

## Results Format

Output saved as `.npy` files containing:

```python
{
    step_num: {
        'rankme': float,
        'alpha': float,
        'eigenspectrum': array,
        'r2': float
    }
}
```

## Citation

```bibtex
@article{li2025tracing,
  title={Tracing the Representation Geometry of Language Models from Pretraining to Post-training},
  author={Li, Melody Zixuan and Ghosh, Arna and Kumar, Sham M. and Kawaguchi, Kenji and Precup, Doina and Lajoie, Guillaume and Bengio, Yoshua},
  journal={arXiv preprint arXiv:2509.23024},
  year={2025}
}
```

## License

MIT License
