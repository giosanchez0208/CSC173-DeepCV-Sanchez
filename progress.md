# CSC173 Deep Computer Vision Project Progress Report
**Student:** Gio Kiefer A. Sanchez, 2022-0025
**Date:** December 31, 2025
**Repository:** [https://github.com/giosanchez0208/CSC173-DeepCV-Sanchez](https://github.com/giosanchez0208/CSC173-DeepCV-Sanchez)

## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 20,000 synthetic images generated with pixel-level masks |
| Initial Training | ✅ Completed | Base model trained for 68 epochs on YOLO11n-seg |
| Baseline Evaluation | ✅ Completed | Compared against EasyOCR and Pytesseract |
| Model Fine-tuning | ✅ Completed | Refinement phase completed (40 epochs) with progressive unfreezing |

## 1. Dataset Progress
* **Total images:** 20,000 (with segmentation masks)
* **Train/Val/Test split:** 80% / 10% / 10%
* **Classes implemented:** 36 (A-Z, 0-9)
* **Preprocessing/Augmentations applied:** * **Text-Level:** Paint chipping via erosion masks and parametric embossing
    * **Plate-Level:** Dirt/grime overlays and auxiliary text placement
    * **Geometric:** 4-point perspective warp and bicubic rotation
    * **CCTV Simulation:** Motion blur kernels, JPEG compression artifacts, and aggressive sharpening
    * **ISP Simulation:** CLAHE on LAB color space and grayscale conversion

## 2. Training Progress

**Current Metrics (Custom OCR Model):**
| Metric | Value |
|--------|-------|
| Precision | 0.0343 |
| Recall | 0.0104 |
| mAP@0.5 | 0.0078 |
| Inference Time | 15.9ms |

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| Visually confusable characters | ✅ Fixed | Implemented `SimilarityAwareTopKLoss` to reduce penalties for similar pairs (O/0, I/1) |
| Digit-heavy class imbalance | ✅ Fixed | Developed `CharacterBalancer` and 50% letter-heavy random formats |
| Hardware constraints | ⏳ Ongoing | Limited training to 68 epochs (base) and 40 epochs (refinement) due to Colab quotas |
| Sim-to-real gap | ⏳ Ongoing | Added motion blur, JPEG artifacts, and real-world background compositing |

## 4. Next Steps (Before Final Submission)
- [ ] Attempt final training run if dedicated GPU resources are secured
- [ ] Implement Character Error Rate (CER) and Word Error Rate (WER) metrics
- [ ] Validate model against real-world Philippine CCTV footage
- [ ] Complete final README with failure mode analysis