<div align="center">


[![TMLR 2026](https://img.shields.io/badge/TMLR-2026-8A2BE2)](https://openreview.net/forum?id=5ChJTFLi5r)
[![arXiv](https://img.shields.io/badge/arXiv-2610.13510-B31B1B?logo=arxiv)](https://arxiv.org/abs/2610.13510)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](https://arxiv.org/pdf/2610.13510)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=5ChJTFLi5r)
[![Project Page](https://img.shields.io/badge/Project-Website-1572B6?logo=googlechrome&logoColor=white)](https://hamzashafiq28.github.io/projects/PRISM/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/hamzashafiq28/PRISM)

# PRISM

### Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis

**Hamza Shafiq · Jialu Tang · Ting Dang · Jun Hu · Aaqib Saeed**

<img src="figures/prism_animation.gif" alt="PRISM multimodal clinical prognosis method animation" width="860" />

*Accepted at Transactions on Machine Learning Research (TMLR), 2026.*

</div>

## Overview

**PRISM** (*Population-Retrieved Inference for Structured Multimodal Prognosis*) combines structured clinical measurements, ECG waveforms, longitudinal patient history, and population-level retrieval to support clinical prognosis. It uses learned representations of similar training encounters rather than placing individual retrieved records into the language-model prompt.

For the method, full experiments, and limitations, see the **[paper](https://arxiv.org/abs/2610.13510)** or the **[project website](https://hamzashafiq28.github.io/projects/PRISM/)**.

## Code Structure

| Path | Purpose |
| --- | --- |
| [`llava/model/multimodal_encoder/`](llava/model/multimodal_encoder/) | Multimodal ECG and EHR encoders |
| [`llava/model/language_model/`](llava/model/language_model/) | Language-model integration |
| [`llava/train/evaluate.py`](llava/train/evaluate.py) | Main evaluation implementation |
| [`llava/train/train_stage1.py`](llava/train/train_stage1.py) | Stage-1 training utility |
| [`llava/train/train_stage2.py`](llava/train/train_stage2.py) | Stage-2 training utility |
| [`prism.sh`](prism.sh) | Evaluation launcher |
| [`figures/`](figures/) | Method animation and figures |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hamzashafiq28/PRISM.git
cd PRISM
```

### 2. Install dependencies

Set up an appropriate Python environment, then install the repository requirements:

```bash
pip install -r requirements.txt
```

### 3. Configure paths and run evaluation

`prism.sh` launches `llava/train/evaluate.py`. Provide paths to your model checkpoint, FAISS index, and prepared clinical data:

```bash
export CHECKPOINT=/path/to/checkpoint
export FAISS_DIR=/path/to/faiss_index
export TEST_JSON=/path/to/test_raft2.json
export TRAIN_VAL_CSV=/path/to/mds_ed_train.csv
export TEST_CSV=/path/to/mds_ed_test.csv
export PATIENT_JSON=/path/to/patient_data_96h.json
export H5_ROOT=/path/to/mimic_h5
export OUTPUT_DIR=/path/to/output_eval_dir

bash prism.sh
```

**Required environment variables:** `CHECKPOINT`, `FAISS_DIR`, `TEST_JSON`, `TRAIN_VAL_CSV`, `TEST_CSV`, `PATIENT_JSON`, `H5_ROOT`, and `OUTPUT_DIR`.

**Optional settings:**

- `ECG_CACHE`: path to an ECG cache (default: empty).
- `MEDGEMMA_MODEL`: model identifier (default: `google/medgemma-4b-it`).

Evaluation outputs are written to `OUTPUT_DIR`, including predictions, summaries, and analysis artifacts. This release requires separately prepared data, checkpoint files, and retrieval assets; they are not downloaded automatically by the script.

## Citation

If you find PRISM useful in your research, please cite the TMLR paper:

```bibtex
@article{shafiq2026prism,
  title   = {Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis},
  author  = {Shafiq, Hamza and Tang, Jialu and Dang, Ting and Hu, Jun and Saeed, Aaqib},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=5ChJTFLi5r}
}
```

---

<div align="center">

**[Paper](https://arxiv.org/abs/2610.13510) · [OpenReview](https://openreview.net/forum?id=5ChJTFLi5r) · [Project Page](https://hamzashafiq28.github.io/projects/PRISM/) · [GitHub](https://github.com/hamzashafiq28/PRISM)**

If this repository helps your research, consider giving it a ⭐.

</div>
