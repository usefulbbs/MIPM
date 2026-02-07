# MIPM — Mechanism-Guided Interpretable Physical Modeling (Coal NIR)

This repository provides the implementation for our paper on **mechanism-guided interpretable modeling** for coal quality evaluation using **multimodal near-infrared (NIR) spectra** measured at multiple sampling distances.  
The goal is to **stably identify physically meaningful *continuous spectral band intervals*** by integrating:

- **Correlation-based analysis** (e.g., Kendall, Spearman, dCor, MIC, MI)
- **Model-driven importance estimation** (e.g., Random Forest MDI/MDA, XGBoost, Linear Regression)
- **PLS + VIP + stability analysis** (multi-view / multi-distance stability)

> Key informative bands are defined as spectral regions that consistently achieve high importance scores across multiple analytical methods, and whose robustness is confirmed via stability analysis.

---

## Overview

### What this code does
1. Loads NIR spectral datasets acquired at different sampling distances (treated as different **views/modalities**).
2. Computes feature importance rankings using:
   - correlation-based metrics,
   - model-driven methods,
   - PLS–VIP + resampling-based stability.
3. Produces **integrated rankings** (e.g., weighted fusion of model-driven + correlation).
4. Converts discrete ranked features into **continuous wavelength intervals** (final band selection).
5. Validates effectiveness via **5-fold cross-validation**, reporting metrics such as **MSE / R²**.

### Why multimodal (multi-distance) matters
Spectra at different distances differ due to geometric optical-path changes and diffuse reflectance effects. The framework emphasizes **cross-view consistency** to improve robustness under measurement uncertainty.

---
