# Nature Communications Reporting Summary

This document addresses the key items from the Nature Communications Reporting Summary checklist.

## Statistical Parameters

### Sample size
- Each experimental configuration was repeated **n = 30** times with deterministic random seeds (seed = run number).
- Total: **4,500 experimental runs** across 3 hardware platforms, 6 models, and 76 configurations.
- No statistical method was used to predetermine sample size. We chose n=30 based on the central limit theorem for robust mean/standard deviation estimation.

### Data exclusions
- **6 records** from `exp2_image_extra_Mac.json` were excluded due to missing `total_energy_j` values (incomplete power measurement runs). No other exclusions.

### Replication
- All experiments used deterministic random seeds and fixed prompts, ensuring full reproducibility.
- Image/video generation: Coefficient of variation (CV) 0.6–3.4% across platforms, indicating high reproducibility.
- Text generation: Higher CV (1.6–19.5%) due to variable output lengths from early EOS tokens, which is inherent to autoregressive generation with greedy decoding.

### Randomization
- Not applicable (this is a measurement/benchmarking study, not a controlled experiment with treatment groups).

### Blinding
- Not applicable (hardware platforms are the independent variable and cannot be blinded).

## Statistical Tests
- **Welch's t-test** (two-sided, unequal variances): Used for all platform comparisons. All reported comparisons have p < 10⁻²⁰.
- **Cohen's d**: Effect sizes reported (all > 5, indicating very large effects).
- **Power-law fitting**: E = α·N^β fitted via nonlinear least squares (Levenberg-Marquardt algorithm, SciPy 1.11). Goodness of fit assessed via R².
- **Crossover points**: Determined by linear interpolation of energy ratios between adjacent configurations.

## Data Availability
- All 4,500 experimental records (JSON format) will be deposited at [GitHub repository URL].
- Source data for all figures and tables are provided as Source Data files.

## Code Availability
- Measurement scripts (Python): Mac experiment script, Colab A100/H100 notebooks.
- Analysis and figure generation scripts (Python, matplotlib).
- All code will be available at [GitHub repository URL].

## Software
- Python 3.11+
- PyTorch 2.4.1 (Mac), PyTorch 2.10.0 (A100/H100 via Google Colab)
- CUDA 12.8 (GPU platforms)
- Apple `powermetrics` (Mac energy measurement)
- NVIDIA `pynvml` (GPU energy measurement)
- SciPy 1.11 (curve fitting)
- matplotlib 3.8+ (figure generation)
- OpenCLIP (CLIP score measurement)

## Hardware
| Platform | Processor | Memory | TDP | Energy Tool |
|----------|-----------|--------|-----|-------------|
| Apple Mac mini | M4 Pro (16-core CPU, 20-core GPU) | 24 GB LPDDR5X | 22W | powermetrics (100ms) |
| NVIDIA A100 | A100-SXM4-40GB | 40 GB HBM2e | 400W | pynvml (50ms) |
| NVIDIA H100 | H100-80GB-HBM3 | 80 GB HBM3 | 700W | pynvml (50ms) |

## Models
All models are open-weight and publicly available via Hugging Face:
- microsoft/Phi-3-mini-4k-instruct (3.8B)
- runwayml/stable-diffusion-v1-5 (0.9B)
- stabilityai/stable-diffusion-xl-base-1.0 (3.5B)
- guoyww/animatediff-motion-adapter-v1-5 
- facebook/musicgen-small (589M)
- facebook/musicgen-medium (2.0B)

## Ethical Considerations
- No human subjects were involved.
- No animal subjects were involved.
- This study measures energy consumption of publicly available AI models and does not raise dual-use concerns.
