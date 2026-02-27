# HarDNet-E2
### Enhanced Harmonic Dense Connectivity for Pediatric Renal Ultrasound Classification

HarDNet-E2 is a structured reformulation of HarDNet85 designed for speckle-dominated pediatric renal ultrasound imaging.  
The model introduces shallow-to-deep semantic reinforcement and structured even-layer aggregation with optional squeeze-and-excitation (SE) recalibration and residual fusion.

This repository provides training, evaluation, calibration, and visualization utilities for reproducible medical image classification experiments.

---

## 🔬 Key Contributions

- Structured shallow-to-deep semantic skip reinforcement
- Even-layer aggregation inside HarDBlocks
- Optional SE recalibration and residual fusion
- Confidence calibration analysis (Expected Calibration Error)
- Bootstrap statistical testing
- Grad-CAM visualization for anatomical interpretability

---

## 📊 Problem Setting

Binary or multi-class classification of pediatric renal ultrasound images:

- **Healthy**
- **Hydronephrosis**
- (Optional third class: Other)

Input: ROI-cropped ultrasound images (224×224)  
Output: Class probabilities via softmax

---

## 🏗 Architecture Overview

HarDNet-E2 extends HarDNet85 with:

- Harmonic dense connectivity
- Even-indexed semantic aggregation
- Channel attention (SE blocks)
- Residual fusion inside HarDBlocks

The architecture is implemented in:
