# UKDRI GitHub Training — GitHub Actions for MLOps

**Session date:** 16 April 2026

This is the reference repository for the UKDRI GitHub training session on setting up GitHub Actions for MLOps. It will be used as a live demo during the session — a new workflow `.yml` file will be created from scratch and wired up to this pipeline.

---

## The ML Problem

A logistic regression model trained to predict `status` — whether a subject has Parkinson's disease (`1`) or is healthy (`0`) — from **22 voice acoustic features** (jitter, shimmer, harmonics-to-noise ratio, and related measures).

**Dataset:** [UCI Parkinson's Disease Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons) (195 subjects, open-access).  
Downloaded at runtime by `data/download_data.py`, which falls back to [OpenML](https://www.openml.org/d/1488) if UCI is unavailable.

---

## Project Structure

```
.github/workflows/   — workflow definitions
data/                — dataset download script
src/                 — train.py, evaluate.py, predict.py
tests/               — data integrity and model quality tests
params.yaml          — model hyperparameters
requirements.txt
```

---
