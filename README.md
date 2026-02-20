# WheatGPCPipeline

A machine learning pipeline for predicting **winter wheat grain protein content (GPC)** from Sentinel-2 satellite imagery, meteorological data, and soil properties. The pipeline implements phenology-aware temporal aggregation strategies and systematically evaluates multiple feature engineering, and modeling approaches.

---

## Highlights

- **Phenology-driven temporal alignment** using GCVI peak detection with double logistic curve fitting
- **Six temporal aggregation strategies** (peak-relative biweekly/monthly, phenological stages, calendar-based, custom periods)
- **Multiple feature selection methods** (mRMR, Boruta, PLS-VIP, RFE consensus)
- **Nested cross-validation** (5-outer × 5-inner × 3-repeats) with Random Forest, LightGBM, XGBoost, and ElasticNet
- **Fully configurable** via a single YAML configuration file

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WheatGPCPipeline                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Sentinel-2   │    │ Meteorological│    │   Soil & Topographic    │   │
│  │  Daily VI &   │    │   Variables   │    │     Properties          │   │
│  │  Spectral     │    │  (ERA5-Land)  │    │   (gsSURGO + GEE)    │   │
│  │  Bands (GEE)  │    │    (GEE)      │    │                        │   │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘   │
│         │                   │                        │                  │
│         ▼                   │                        │                  │
│  ┌──────────────┐           │                        │                  │
│  │ GCVI Peak    │           │                        │                  │
│  │ Detection &  │           │                        │                  │
│  │ Smoothing    │           │                        │                  │
│  └──────┬───────┘           │                        │                  │
│         │                   │                        │                  │
│         ▼                   ▼                        │                  │
│  ┌─────────────────────────────────┐                 │                  │
│  │   Temporal Alignment &          │                 │                  │
│  │   Aggregation (6 strategies)    │                 │                  │
│  └──────────────┬──────────────────┘                 │                  │
│                 │                                    │                  │
│                 ▼                                    ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Feature Engineering                            │  │
│  │                          Derivatives
│  └──────────────────────────────┬────────────────────────────────────┘  │
│                                 │                                       │
│                                 ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Feature Selection                              │  │
│  │                   Variance filter · Boruta
│  └──────────────────────────────┬────────────────────────────────────┘  │
│                                 │                                       │
│                                 ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Modeling & Evaluation                          │  │
│  │                Nested CV · RF · LightGBM · XGBoost
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
WheatGPCPipeline/
├── configs/
│   └── config.yaml                  # Pipeline configuration
├── src/                             # Core Python modules
│   ├── config.py                    # Configuration loading & feature type definitions
│   ├── data_loading.py              # Load spectral, meteo, soil, elevation data
│   ├── peak_detection.py            # GCVI peak detection & quality control
│   ├── double_logistic.py           # Double logistic sigmoid curve fitting
│   ├── smoothing.py                 # Savitzky-Golay smoothing
│   ├── temporal_alignment.py        # Temporal aggregation strategies
│   ├── feature_engineering.py       # Derived features
│   ├── feature_selection.py         # Boruta
│   ├── modeling.py                  # Model training with nested cross-validation
│   ├── evaluation.py                # Metrics computation & visualization
│   ├── saxton_rawls.py              # Saxton & Rawls pedotransfer functions
│   └── utils.py                     # Logging & utilities
├── notebooks/                       # Analysis notebooks
├── data/
│   ├── raw/                         # Raw input data (not tracked)
│   └── processed/                   # Processed features (not tracked)
├── models/                          # Trained models & predictions (not tracked)
├── figures/                         # Generated visualizations (not tracked)
├── requirements.txt                 # Python dependencies
└── README.md
```
