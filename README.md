# [Project Title: e.g., Real-Time Object Detection for Waste Sorting]
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** [Your Full Name], [Student ID]  
**Semester:** [e.g., AY 2025-2026 Sem 1]  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
150-250 words: Summarize problem (e.g., "Urban waste sorting in Mindanao"), dataset, deep CV method (e.g., YOLOv8 fine-tuned on custom trash images), key results (e.g., 92% mAP), and contributions.

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Standard Optical Character Recognition (OCR) models and off-the-shelf text detectors are not designed around license plate fonts and layouts, and they typically require heavy preprocessing to cope with blur, noise, and difficult lighting in unconstrained road scenes. [1][2] These limitations are amplified in the Philippine context, where legacy and modern plate designs coexist, diverging from the datasets on which generic OCR systems are trained. [9] This project aims to build a Philippine-specific license plate character recognizer by training a segmentation-based model on synthetic images that replicate local plate styles, backgrounds, and capture artifacts — leveraging pretrained backbones or Ultralytics segmentation models where appropriate to accelerate convergence and improve sim-to-real performance. [4][7]

### Objectives
- Design and implement a procedural data generation pipeline that synthesizes photorealistic Philippine license plate images (private, public, and motorcycle formats) with automatic pixel-level character mask annotations.
- Build and/or fine-tune a semantic segmentation model (custom or pretrained) to detect and classify individual alphanumeric characters on license plates under varied imaging conditions, specifically targeting the main license plate number while ignoring auxiliary text (e.g., region codes, small print); Ultralytics segmentation tooling may be leveraged for experiments and baselines.
- Achieve robust recognition performance across diverse lighting, motion blur, emboss effects, and background clutter, targeting high character-level accuracy on held-out synthetic and real test sets.

![Problem Demo](images/problem_example.gif) [web:41]

## Related Work
- [Paper 1: YOLOv8 for real-time detection [1]]
- [Paper 2: Transfer learning on custom datasets [2]]
- [Gap: Your unique approach, e.g., Mindanao-specific waste classes] [web:25]

## Methodology
### Dataset
**Source:** Fully synthetic dataset generated via custom procedural pipeline (`plate_generator.ipynb`)  
**Size:** 20,000 synthetic Philippine license plate images (512x512 px)  
**Split:** 80/10/10 train/val/test (16,000/2,000/2,000 images)

#### Plate Styles
The dataset replicates authentic Philippine LTO plate designs sourced from [Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_Philippines):
- **Vehicle plates** (private/passenger): White/green, yellow/black, white/red, white/blue backgrounds
- **Motorcycle plates**: White/black, white/pink, white/green with colored detail rectangles
- **Beveled motorcycle plates**: Top-corner beveled edges with small auxiliary text
- **Government plates**: Blue/black scheme
- **Rizal commemorative plates**: Custom background image with green text overlay

#### Character Distribution & Balancing
To ensure uniform class representation across all 36 classes (A-Z, 0-9):
- **Adaptive format selection:** 50% follow standard 3-letter + 4-digit format (e.g., `ABC 1234`); 50% use letter-heavy random formats (4-7 letters, 1-3 digits) to counterbalance digit-heavy standard plates
- **Dynamic balancing:** Custom `CharacterBalancer` class tracks per-character usage in real-time and adjusts sampling weights to minimize distribution variance
- **Target ratio:** 2.6 letters per digit (26 letters / 10 digits), achieved via continuous monitoring of cumulative letter/digit usage

#### Synthetic Augmentations
**Text-Level Obscurations (applied during plate rendering):**
- **Paint chipping** (50% probability): Random erosion masks (0.3-0.7x letter size) applied per character to simulate wear
- **Emboss effects** (100%): Randomized light-source angles (0-360 degrees) and strength (0-3.5 px) create realistic raised-text appearance
- **Small auxiliary text** (70%): Dummy region codes/stickers placed at top/bottom using scaled fonts

**Plate-Level Obscurations:**
- **Dirt/grime overlays** (25% probability): Alpha-blended texture masks confined to plate area via pixel-level masking

**Image-Level Augmentations (applied post-rendering):**
- **Perspective warp** (50% probability): Random 4-point homography with max displacement 0.05-0.25x plate dimensions
- **Rotation** (40%): +/- 5 to +/- 20 degrees with bicubic resampling on transparent canvas
- **Motion blur** (30%): Directional kernel (10-25 px length, random angle, 5-15 samples)
- **JPEG compression artifacts** (60%): Quality 10-40 with 4:2:0 chroma subsampling + scanline darkening to mimic surveillance footage
- **CLAHE** (35%): Clip limit 1.5-4.0 on LAB color space to simulate ISP auto-exposure
- **Aggressive sharpening** (25%): Unsharp mask (strength 1.5-3.5) creating halo effects typical of cheap CCTV cameras
- **Grayscale conversion** (20%): Preserves alpha channel for mixed-color datasets

**Background Integration:**
- 80% probability: Plates composited onto random real-world backgrounds (roads, walls, textures) from `images/image_backgrounds/`
- Random plate scaling (0.7-1.0x) and placement within 512x512 canvas
- Final split: 30% clean images, 70% with full augmentation pipeline

#### Annotation Format
**YOLO Segmentation Format:** Each character annotated as a separate polygon instance
```
<class_id> x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
```
- Class IDs: A-Z -> 0-25, 0-9 -> 26-35
- Polygon extraction: `cv2.findContours()` on per-character binary masks (minimum 3 points, area >1 px)
- Mask transformations: All geometric augmentations (rotation, perspective) applied identically to both images and annotation masks to maintain spatial consistency

#### Preprocessing
- **No preprocessing required:** Model ingests raw 512x512 RGBA images directly
- **Training-time augmentations** (via Ultralytics): Mosaic, HSV jitter, horizontal flip (standard YOLO augmentations applied on top of synthetic variations)

![Sample Plates](documentation/sample_plates.jpg)  
*Figure: Representative samples from license plate dataset*

### Architecture
**Base Model:** YOLO11n-seg (Ultralytics Segmentation)  
**Backbone:** CSPDarknet with C2f blocks  
**Neck:** Path Aggregation Network (PANet)  
**Heads:** 
- Detection head (bounding boxes)
- Segmentation head (polygon masks)
- Classification head (character recognition with custom similarity-aware loss)

**Custom Components:**
- `SimilarityAwareTopKLoss`: Reduces penalties for visually confusable character pairs (O/0, I/1/L, S/5, etc.) by evaluating top-k predictions against a hand-crafted similarity matrix
- `DynamicSimilarityMatrix`: Updates similarity scores during training using exponential moving average of observed confusion patterns
- `ImprovedSimilarityAwareTopKLoss`: Adds temperature annealing (1.0 to 0.5) and confidence-based adaptive weighting between cross-entropy and similarity-aware components
- `CustomSegmentationTrainer`: Extends YOLO trainer with multi-task loss balancing (40% mask, 30% box, 30% classification) and OCR-specific metrics (CER, WER, top-k accuracy)

**Character Set:** 36 classes (A-Z = 0-25, 0-9 = 26-35)

**Similarity Groups (base_sim=0.6):**
```
O/0/Q, I/1/L, S/5, Z/2, B/8, D/0, G/C, U/V, P/R
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 68 (viable amount given time constraint) | Extended training for fine-grained character features |
| Batch Size | 16 | Balanced GPU memory usage (MPS/CUDA) |
| Image Size | 224 | Matches character-crop resolution |
| Optimizer | SGD | Proven stability for YOLO architectures |
| Base Learning Rate | 0.01 | Standard YOLO starting point |
| Final Learning Rate | 0.01 | Cosine annealing maintains exploration |
| Momentum | 0.937 | YOLO default for SGD convergence |
| Weight Decay | 5e-4 | Regularization without over-penalizing |
| Warmup Epochs | 3.0 | Gradual ramp-up prevents early instability |
| Device | MPS (Metal) / CUDA / CPU | Auto-detected for M4 Mac or GPU systems |

**Augmentation Strategy:**
```python
# Color variations (CCTV lighting conditions)
hsv_h: 0.015  # Hue shift
hsv_s: 0.7    # Saturation variation
hsv_v: 0.4    # Brightness/contrast

# Occlusion simulation
erasing: 0.4  # Random erasing (dirt, damage)

# Disabled augmentations (text-specific)
fliplr: 0.0      # No horizontal flip (breaks text orientation)
mosaic: 0.0      # No mosaic (detection-centric augmentation)
mixup: 0.0       # No mixup (interferes with character boundaries)
copy_paste: 0.0  # No copy-paste (mask artifacts)
```

**Loss Function Mechanics:**
- **Base Component (70% weight):** Standard cross-entropy loss for sharp class boundaries
- **Top-k Component (30% weight):** Similarity-weighted penalty on top-2 predictions
    - If model predicts O but target is 0: penalty = (1 - 0.6) = 0.4
    - If model predicts X but target is 0: penalty = (1 - 0.0) = 1.0
- **Temperature Annealing:** Softmax temperature decreases from 1.0 to 0.5 over 300 epochs, transitioning from soft exploration to hard exploitation
- **Adaptive Weighting:** Low-confidence predictions (< 0.5 max probability) increase reliance on similarity-aware loss; high-confidence predictions trust standard cross-entropy

**Training Code Excerpt:**
```python
# Initialize model with custom trainer
model = YOLO('yolo11n-seg.pt')
model.trainer = CustomSegmentationTrainer

# Configure similarity-aware loss
character_loss_fn = ImprovedSimilarityAwareTopKLoss(
        num_classes=36,
        similarity_matrix=similarity_matrix,
        k=2,
        initial_temperature=1.0,
        base_weight=0.7,
        topk_weight=0.3,
        epochs=300
)

# Train with multi-task loss balancing
results = model.train(
        data='dataset/data.yaml',
        epochs=300,
        batch=16,
        imgsz=224,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        device='mps',  # Auto-detected
        project='philippine_lp_ocr',
        name='seg_with_similarity_loss'
)
```

**OCR-Specific Validation Metrics:**
- **Character Error Rate (CER):** Percentage of individual character misclassifications
- **Word Error Rate (WER):** Percentage of full plates with any character error
- **Top-2/3 Accuracy:** Correct character appears in top-k predictions
- **Similarity-Aware Accuracy:** Partial credit using similarity scores (e.g., O vs 0 = 60% credit)

**Early Stopping:**
- Monitors fitness score (combined mAP and loss metric)
- Patience: 50 epochs without improvement
- Saves best checkpoint to `models/custom_ocr_best.pt`

### Refinement Configuration

**Objective:** Reduce classification loss and improve character differentiation for confusable pairs (Q/0/O, L/1/I) through progressive layer unfreezing and enhanced similarity-aware loss.

**Base Model:** `custom_ocr_last.pt` (68-epoch pretrained checkpoint)  
**Total Refinement Epochs:** 40  
**Training Strategy:** 3-phase progressive unfreezing

| Phase | Epochs | Trainable Layers | Learning Rate | Focus |
|-------|--------|------------------|---------------|-------|
| **Phase 1** | 1-12 | Classification head only | 0.005 → 0.0001 | Escape local minimum, retrain classifier |
| **Phase 2** | 13-24 | + Segmentation head | 0.005 → 0.0001 | Balance mask/class predictions |
| **Phase 3** | 25-40 | All layers (full) | 0.005 → 0.0001 | Fine-tune entire network |

**Refined Loss Function:**
```python
RefinedSimilarityAwareTopKLoss(
    num_classes=36,
    similarity_matrix=updated_groups,  # Removed Q from O/0 group
    k=3,                               # Evaluate top-3 predictions
    initial_temperature=0.5,           # Sharper predictions (0.5 → 0.3)
    base_weight=0.5,                   # Balanced CE/similarity weighting
    topk_weight=0.5,
    epochs=40
)
```

**Key Changes:**
- **Similarity Groups Updated:** Q, L removed from O/0/Q and I/1/L groups to increase differentiation penalty
- **Cyclic Learning Rate:** Higher initial LR (0.005 vs 0.001) to shake model from plateau
- **Aggressive Augmentations:** Increased HSV variation (h=0.02, s=0.8, v=0.5), random erasing (0.5), geometric transforms (±5° rotation, 2° shear)
- **Temperature Annealing:** More aggressive (0.5 → 0.3) for sharper class boundaries
- **Adaptive Weighting:** Phase-dependent loss composition (Phase 1: 70% classification, Phase 2: 50%, Phase 3: 30%)

**Checkpoint Management:**
- **Metrics CSV:** Saved to Google Drive every epoch (fast, ~1 second)
- **Model Checkpoints:** 
  - Best model saved immediately when validation loss improves (async upload)
  - Backup saved every 10 epochs to prevent corruption
  - All saves run asynchronously to avoid blocking training
- **Resume Support:** Automatic detection of previous training sessions via `training_progress.csv`

**Performance Target:** Classification loss < 0.35 (from baseline 0.4321 at epoch 68)

**Training Environment:**
- **Storage:** Dataset copied from Google Drive to Colab local SSD via zip extraction (fast I/O)
- **Device:** Auto-detected (CUDA/MPS/CPU)
- **Precision:** Mixed precision (AMP) enabled for faster training

## Experiments & Results
### Metrics
| Model | mAP@0.5 | Precision | Recall | Inference Time (ms) |
|-------|---------|-----------|--------|---------------------|
| Baseline (YOLOv8n) | 85% | 0.87 | 0.82 | 12 |
| **Ours (Fine-tuned)** | **92%** | **0.94** | **0.89** | **15** |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)] [web:41]

## Discussion
- Strengths: [e.g., Handles occluded trash well]
- Limitations: [e.g., Low-light performance]
- Insights: [e.g., Data augmentation boosted +7% mAP] [web:25]

## Ethical Considerations
- Bias: Dataset skewed toward plastic/metal; rural waste underrepresented
- Privacy: No faces in training data
- Misuse: Potential for surveillance if repurposed [web:41]

## Conclusion
[Key achievements and 2-3 future directions, e.g., Deploy to Raspberry Pi for IoT.]

## Installation
1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-YourLastName`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
torch>=2.0
ultralytics
opencv-python
albumentations

## References
[1] Jocher, G., et al. "YOLOv8," Ultralytics, 2023.  
[2] Deng, J., et al. "ImageNet: A large-scale hierarchical image database," CVPR, 2009. [web:25]

## GitHub Pages
View this project site: [https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/](https://jjmmontemayor.github.io/CSC173-DeepCV-Montemayor/) [web:32]

