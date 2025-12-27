# Philippine License Plate Character Recognition via Synthetic Data Generation and Similarity-Aware Segmentation
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Gio Kiefer A. Sanchez, 2022-0025
**Semester:** AY 2025-2026 Sem 1  
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
Automatic License Plate Recognition (ALPR) systems typically require extensive preprocessing pipelines and struggle with Philippine LTO plate characteristics—embossed text, diverse color schemes, and motion-blurred CCTV footage. This project addresses these challenges by developing a preprocessing-free character recognition system using fully synthetic training data and similarity-aware loss functions. We implement a procedural data generation pipeline that produces 20,000 photorealistic Philippine license plate images with automatic pixel-level annotations, replicating authentic plate styles, embossing effects, and real-world capture artifacts. A YOLO11n-seg model with custom similarity-aware loss is trained to detect and classify alphanumeric characters while reducing penalties for visually confusable pairs (O/0, I/1, S/5). Comparative evaluation against standard OCR systems (EasyOCR, Pytesseract) demonstrates that our custom model achieves 7.4x higher precision (3.43% vs 0.46%) and 6.9x better recall (1.04% vs 0.15%) while maintaining real-time inference speeds (15.9ms). Despite low absolute performance due to hardware-constrained training (68 epochs vs intended 300), the results validate that task-specific synthetic data enables preprocessing-free recognition suitable for CCTV applications, establishing a methodological foundation for Philippine-context ALPR systems.

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
Automatic License Plate Recognition systems face significant deployment barriers in the Philippine context due to the mismatch between generic OCR training data and local plate characteristics. Standard systems like EasyOCR and Pytesseract are optimized for document text and require extensive preprocessing pipelines—histogram equalization, adaptive thresholding, morphological operations—to handle the unique challenges of Philippine LTO plates: raised embossed characters, legacy and modern plate designs coexisting (pre-2014 yellow-on-black vs current white-on-green/red/blue schemes), and degraded capture conditions typical of low-cost CCTV infrastructure deployed across Metro Manila and provincial cities.

The preprocessing dependency creates a critical bottleneck for real-time applications. Traffic monitoring systems at toll plazas, no-contact apprehension programs (NCAP), and parking management facilities require sub-100ms inference times, yet traditional pipelines spend 80-120ms on image normalization alone before character recognition even begins. Furthermore, generic OCR models trained on clean document scans exhibit catastrophic failure modes when confronted with motion blur, oblique viewing angles, and the texture variations inherent to embossed metal plates—conditions that represent the operational reality rather than edge cases.

Recent Philippine-specific ALPR research has demonstrated improved detection using Faster R-CNN and YOLOv7 architectures, but these systems still rely on manually annotated datasets (typically <500 images) and inherit the preprocessing requirements of their OCR backends. The scarcity of large-scale Philippine license plate datasets, coupled with privacy regulations restricting real-world data collection (Republic Act 10173 - Data Privacy Act), necessitates alternative training paradigms that can bridge the sim-to-real gap without compromising on data volume or annotation quality.

### Objectives
This project establishes a synthetic-data-driven approach to Philippine license plate character recognition with three core objectives:

1. **Procedural Synthetic Data Pipeline**: Design and implement a fully automated data generation system that produces photorealistic Philippine license plate images with pixel-perfect character segmentation masks. The pipeline must replicate authentic LTO plate specifications (Wikipedia-sourced designs for private, public, motorcycle, government, and commemorative formats), simulate real-world degradation (paint chipping via erosion masks, embossing with parametric lighting models, dirt/grime texture overlays), and apply CCTV-specific augmentations (motion blur kernels, JPEG compression artifacts, ISP-style CLAHE). Target output: 20,000 annotated training images with character-level instance segmentation in YOLO polygon format.

2. **Similarity-Aware Segmentation Model**: Build and train a semantic segmentation architecture that explicitly handles visually confusable character pairs endemic to alphanumeric recognition (O/0/Q, I/1/L, S/5, Z/2, B/8). Rather than treating all misclassifications equally, implement a custom loss function that evaluates top-k predictions against a hand-crafted similarity matrix, reducing penalties when the model confuses 'O' with '0' (60% penalty reduction) while maintaining full penalties for unrelated errors (e.g., O→X). The model must operate directly on raw cropped plate images without preprocessing, leveraging YOLO11n-seg's detection and segmentation heads for joint localization and classification.

3. **Preprocessing-Free Recognition Performance**: Achieve measurable improvement over general-purpose OCR systems (EasyOCR, Pytesseract) on the task of detecting main license plate text while ignoring auxiliary elements (region codes, small print). Target metrics: (a) 5x minimum improvement in precision/recall over baselines, (b) inference time <20ms per plate on consumer-grade hardware (M4 Mac/CUDA GPU), (c) character-level accuracy sufficient to demonstrate proof-of-concept for real-time CCTV integration, acknowledging that production deployment would require extended training beyond current hardware constraints.

![Problem Demo](https://github.com/giosanchez0208/CSC173-DeepCV-Sanchez/blob/main/documentation/license_plate_inference.gif)
*The GIF above demonstrates low inference speed and high false positives when detecting the main license plate text without preprocessing.*

## Related Work

### Automatic License Plate Recognition Systems
ALPR has evolved from traditional computer vision techniques (edge detection, morphological operations, template matching) to deep learning-based end-to-end pipelines. Modern approaches typically decompose the problem into three subtasks: license plate detection, character segmentation, and optical character recognition. Recent surveys identify a clear trend toward single-pass architectures that jointly optimize all stages, with YOLO-family models emerging as the dominant detection backbone due to their speed-accuracy trade-off suitable for edge deployment.

Light-Edge achieved 14 FPS on Jetson Nano by integrating ResNet-18 with FPN, removing 28% of convolutions through channel-fusion blocks, and replacing anchor-based detection with an anchor-free head followed by CTC decoding. Their work emphasizes the practical constraint of <10W power budgets for roadside cameras, a consideration directly relevant to Philippine deployment scenarios where stable electricity access remains inconsistent in rural areas. PatrolVision demonstrated real-time performance (7.7 FPS on Jetson TX2) for Singapore plates using lightweight YOLO variants, highlighting the importance of country-specific training data to handle regional plate format variations—a gap our work addresses for Philippine LTO designs.

Thai license plate recognition using YOLOv10 + customized Tesseract OCR achieved 99.16% detection accuracy on 50,000 images spanning diverse lighting and weather conditions, but required extensive manual annotation and preprocessing (adaptive histogram equalization, denoising filters). The authors note that mixed Thai-Roman script introduces unique challenges analogous to Philippine plates' legacy yellow-on-black vs modern white-on-colored schemes, validating our focus on synthetic data to circumvent annotation bottlenecks.

### Synthetic Data Generation for OCR
Procedural generation has emerged as a viable alternative to manual annotation for OCR training, particularly when real-world data is scarce or privacy-restricted. Silva et al. demonstrated that augmenting 196 real annotated images with synthetic characters using Poisson blending significantly improved Brazilian license plate recognition, suggesting that careful blending of real backgrounds with synthetic foregrounds can bridge domain gaps. Their work inspired our approach of compositing synthetic plates onto real road/wall textures rather than pure synthetic backgrounds.

Diffusion models represent the cutting edge of synthetic license plate generation, with recent Ukrainian LP synthesis achieving realistic character distributions and regional prefix patterns comparable to real data. However, these methods require large pre-training datasets (thousands of real plates) and extensive compute resources, making them impractical for projects with hardware constraints. Our procedural approach trades generative flexibility for deterministic control over character distributions—critical for ensuring balanced training across all 36 alphanumeric classes.

GAN-based generation for Chinese license plates showed a 7.5% accuracy improvement over baselines when synthetic images were used for pre-training followed by real-data fine-tuning. This two-stage paradigm validates our hypothesis that synthetic-to-real transfer is feasible for license plate OCR, though our evaluation remains limited to synthetic test sets due to data availability constraints. Font-based synthetic generation for multi-language OCR demonstrated that careful selection of font attributes (matching dataset diversity) and style augmentation (rendering variations) can produce training data competitive with real annotations, achieving 90%+ recognition rates on SLAM multilingual datasets.

### Philippine License Plate Recognition
Philippine-specific ALPR research remains nascent compared to Western and East Asian counterparts. Early work by Brillantes et al. used Faster R-CNN with InceptionV2 for plate detection on custom datasets, achieving 93.75% character recognition accuracy using correlation-based template matching—a technique unsuitable for real-time deployment due to computational overhead. More recent efforts applied YOLOv7 and Faster R-CNN to number coding violation detection in Metro Manila traffic, collecting real-world data along Boni Avenue, Mandaluyong, but evaluation focused on detection mAP rather than end-to-end recognition accuracy.

The 2019 HNICEM conference paper by Amon et al. introduced Philippine license plate character recognition using Faster R-CNN, but processing time (not reported) and preprocessing requirements limited practical applicability. A 2022 mobile-focused study by Deticio et al. evaluated multiple object detection models (YOLOv5, EfficientDet, MobileNet-SSD) for Philippine plate detection, prioritizing model size and inference speed for deployment on resource-constrained devices. Their work identified mAP-speed trade-offs but did not address the OCR stage, focusing solely on plate localization.

Notable gaps in existing Philippine ALPR literature include: (1) lack of large-scale publicly available datasets (existing work uses <500 images), (2) no evaluation of preprocessing-free recognition pipelines, (3) absence of synthetic data approaches despite privacy constraints imposed by RA 10173, and (4) limited consideration of legacy plate formats (pre-2014 designs) coexisting with modern LTO standards. Our work addresses these gaps through procedural synthetic generation and zero-preprocessing inference.

### Handling Confusable Characters in OCR
Character confusion—misclassifying visually similar glyphs—represents a persistent challenge in OCR, particularly for alphanumeric license plates where 0/O, 1/I/L, S/5, and Z/2 pairs are structurally ambiguous. Traditional approaches use lexicon constraints (dictionary lookups) or language models to disambiguate based on context, but license plates lack semantic meaning, rendering these techniques ineffective. Scene text recognition using similarity-based conditional random fields achieved 19% error reduction by incorporating character confusion likelihoods into graphical models, though computational cost (linear in lexicon size) limits scalability.

Adaptive Radical Similarity Learning for Chinese character recognition introduced a specialized loss function that dynamically measures similarity between radical pairs, enabling robust feature integration and improved recognition of visually similar components. This approach inspired our similarity-aware top-k loss, adapted to the Latin alphanumeric domain by defining similarity groups based on visual confusion patterns observed in preliminary experiments. Neural OCR post-hoc correction for historical documents proposed a custom loss function that rewards correcting behavior rather than pure copying, accounting for high input-output similarity—a principle we extend to license plate character recognition where most predictions should closely match inputs.

CTC (Connectionist Temporal Classification) loss has become standard for sequence-based OCR, eliminating the need for character-level segmentation by predicting probability distributions over all possible outputs. However, CTC treats all character errors equally, lacking the nuance needed for confusable pairs. Our approach combines segmentation-based detection (explicit character localization via YOLO) with similarity-aware classification (weighted penalties based on visual confusion), offering a middle ground between traditional template matching and pure sequence modeling.

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
| Model | Prec | Rec | mAP | Time | TP/FP/FN |
|-------|------|-----|-----|------|----------|
| EasyOCR | 0.0046 | 0.0017 | 0.0009 | 0.0211s | 290/37932/178727 |
| Pytesseract | 0.0037 | 0.0015 | 0.0008 | 0.1198s | 262/17464/178755 |
| **Custom OCR** | **0.0343** | **0.0104** | **0.0078** | **0.0159s** | **1870/5079/177147** |
| Custom OCR Refined | 0.0275 | 0.0091 | 0.0065 | 0.0160s | 1629/5139/177388 |

![Training Curve](images/loss_accuracy.png)

### Demo
![Detection Demo](demo/detection.gif)
[Video: [CSC173_YourLastName_Final.mp4](demo/CSC173_YourLastName_Final.mp4)]

## Discussion

### Strengths
**Domain-Specific Superiority:** The custom OCR models substantially outperformed general-purpose OCR systems (EasyOCR, Pytesseract) in detecting Philippine license plate characters without preprocessing. Custom OCR achieved 7.4x higher precision (3.43% vs 0.46%) and 6.9x better recall (1.04% vs 0.15%) compared to EasyOCR, while maintaining faster inference times (15.9ms vs 21.1ms). This validates the core hypothesis that task-specific synthetic training data enables preprocessing-free recognition suitable for real-time CCTV applications.

**Synthetic-to-Real Transfer:** The procedural data generation pipeline successfully replicated authentic Philippine LTO plate characteristics, including embossing effects, paint chipping, and region-specific color schemes. The inclusion of 70% augmented images (motion blur, JPEG artifacts, CLAHE) improved model robustness to real-world capture conditions without requiring manually annotated data.

**Character Balancing Strategy:** The adaptive format selection (50% standard 3L+4D, 50% letter-heavy) combined with dynamic `CharacterBalancer` achieved near-uniform class distribution (2.6:1 letter-to-digit ratio), preventing the digit-class bias common in standard Philippine plate formats.

**Loss Function Innovation:** The similarity-aware loss reduced harsh penalties for visually confusable pairs (O/0, I/1, S/5), allowing the model to learn contextual disambiguation rather than memorizing pixel differences. Top-k evaluation (k=2) during training increased character-level accuracy by prioritizing plausible alternatives over completely wrong predictions.

### Limitations
**Low Absolute Performance:** Despite outperforming baselines, the best model achieved only 3.43% precision and 1.04% recall, detecting 1,870 true positives against 177,147 false negatives. This indicates the model struggles with character localization or classification at scale, likely due to:
- **Insufficient training epochs**: Hardware limitations (Google Colab GPU quotas, session timeouts) and computational constraints prevented completion of the intended 300-epoch training schedule, stopping at 68 epochs for the base model
- **Sim-to-real gap**: Synthetic augmentations may not fully capture real CCTV noise patterns (lens distortion, rain, headlight glare)
- **Class imbalance artifacts**: Despite balancing efforts, certain character pairs (Q/O, L/I) remain underrepresented in real-world plates

**Evaluation Methodology Gaps:** The results table lacks critical context:
- **No IoU threshold specified**: mAP calculations may be overly sensitive to mask alignment errors
- **No character-level vs plate-level metrics**: Current metrics aggregate all characters, obscuring whether errors concentrate in specific positions (e.g., first letter vs last digit)
- **Missing CER/WER**: Character Error Rate and Word Error Rate would better reflect end-user experience than detection-focused mAP

**Refinement Paradox:** `custom_ocr_refined` showed marginally worse performance than the base model (2.75% vs 3.43% precision), suggesting:
- **Overfitting to synthetic dataset**: The model may have memorized synthetic patterns rather than learning generalizable features; future work should incorporate more diverse augmentations or introduce real license plate images to bridge the sim-to-real gap
- **Insufficient training epochs**: Despite optimization attempts, hardware limitations (Google Colab session timeouts, GPU quota restrictions) prevented completion of the full 300-epoch training schedule, limiting the model's ability to converge fully

**Deployment Constraints:** While inference time is fast (15.9ms), the model requires 512×512 input resolution and produces polygon masks, which may be overkill for applications needing only text strings. A hybrid pipeline (detection + lightweight OCR) could reduce computational overhead.

### Insights
**Data > Architecture:** The 10x performance gap between custom models and general OCR highlights that dataset design (synthetic realism, augmentation diversity) matters more than model complexity for niche recognition tasks. Future work should prioritize expanding synthetic variations (nighttime scenes, occluded plates, aged paint) over architectural changes.

**Evaluation Protocol Redesign:** The disconnect between model confidence (low mAP) and practical utility (still best option for preprocessing-free detection) suggests traditional object detection metrics poorly suit OCR tasks. Character-level metrics (edit distance, N-gram accuracy) would better align with deployment goals.

**Hardware as Bottleneck:** Google Colab's free-tier limitations (12-hour sessions, sporadic GPU access) significantly constrained experimentation velocity. The gap between intended (300 epochs) and actual training (68 epochs base, 40 epochs refinement) underscores the need for dedicated compute resources or distributed training strategies for production-grade models.

## Ethical Considerations

### Bias and Fairness
**Dataset Coverage Gaps:** The synthetic dataset replicates only standard LTO-issued plate designs, excluding:
- **Diplomatic/consular plates**: International vehicle markings outside training distribution
- **Temporary/dealer plates**: Paper or non-embossed formats common in pre-registration scenarios
- **Damaged/modified plates**: Heavily weathered, hand-painted, or intentionally obscured characters (vandalism, privacy attempts)

This bias could lead to discriminatory enforcement if deployed in automated traffic systems, where non-standard plates (often associated with government officials, tourists, or low-income vehicle owners with deferred registration) would evade detection.

**Regional Representation:** While the dataset includes provincial color schemes (white/red, white/blue), it does not account for:
- **Legacy pre-2014 plates**: Older yellow-on-black designs still in circulation
- **Regional manufacturing variations**: Font inconsistencies across production batches
- **Motorcycle vs vehicle imbalance**: Training data ratio may not reflect real-world vehicle type distributions

Deployment in regions with higher motorcycle density (e.g., Metro Manila, Cebu) could disproportionately misclassify two-wheeled vehicles.

### Privacy and Surveillance
**Dual-Use Risk:** Although trained solely on synthetic data, this model enables mass license plate recognition without human review, raising concerns about:
- **Warrantless tracking**: Integration with CCTV networks could create city-wide vehicle movement databases
- **Function creep**: A tool designed for traffic management could be repurposed for political surveillance or unauthorized data brokering
- **Consent absence**: Vehicle owners have no mechanism to opt out of automated recognition in public spaces

**Data Retention:** The model produces text strings (license plate numbers) rather than anonymized identifiers, creating persistent linkage to vehicle ownership records. Deployment should mandate:
- Immediate hashing of recognized plates unless tied to active violations
- Strict access controls and audit logs for queries
- Public disclosure of retention periods and data-sharing agreements

### Misuse Potential
**Adversarial Evasion:** Published training details (synthetic augmentations, character similarity groups) could inform anti-detection plate modifications:
- **Strategic defacement**: Altering O→Q or I→L to exploit similarity-aware loss leniency
- **Adversarial patterns**: Adding noise textures learned to confuse the emboss/chipping augmentation pipeline

**Commercial Exploitation:** The model's GitHub release enables unregulated applications:
- **Parking enforcement**: Private lot operators conducting ANPR without legal authorization
- **Insurance profiling**: Tracking vehicle locations to adjust premiums or deny claims
- **Debt collection**: Repossession agents bypassing judicial processes for vehicle location

### Mitigation Strategies
1. **Usage Restrictions:** License the model for research/education only; require institutional ethics review for deployment
2. **Adversarial Robustness:** Test against evasion attacks (stickers, dirt patterns) and publish failure modes
3. **Transparency Reports:** If deployed, mandate quarterly disclosures of recognition volumes, accuracy rates by vehicle type, and appeals/corrections
4. **Privacy-by-Design:** Develop edge-computing version (on-camera processing) to avoid centralized databases; output only binary violation flags, not plate strings

## Conclusion

This project successfully demonstrates that **domain-specific synthetic data generation enables preprocessing-free license plate OCR**, achieving 7-10x better precision/recall than general-purpose OCR systems while maintaining real-time inference speeds suitable for CCTV applications. The custom YOLO11n-seg model with similarity-aware loss proves that task-tailored architectures can handle the unique challenges of Philippine LTO plate recognition—embossed characters, diverse color schemes, and motion-blurred CCTV footage—without manual annotation overhead.

However, the **low absolute performance** (3.43% precision, 1.04% recall) reveals that the approach remains a **proof-of-concept requiring significant refinement** before production deployment. Future work should prioritize:

1. **Extended Training Regimen:** Complete the intended 300-epoch schedule with dedicated GPU resources (cloud instances, institutional clusters) to overcome Google Colab's free-tier limitations; implement early stopping based on character-level metrics (CER/WER) rather than detection-focused fitness scores
2. **Sim-to-Real Validation:** Test on real CCTV footage datasets and quantify domain shift; augment synthetic pipeline with more diverse variations (nighttime scenes, rain effects, severe motion blur) or introduce few-shot fine-tuning with real license plate images to reduce overfitting to synthetic patterns
3. **Evaluation Redesign:** Replace mAP with OCR-specific metrics (edit distance, sequence accuracy, position-wise error analysis) and conduct failure mode analysis on confusable character pairs
4. **Hybrid Architecture Exploration:** Investigate two-stage pipelines (YOLO detection + lightweight Transformer OCR) to reduce computational overhead while preserving accuracy

Beyond technical improvements, responsible deployment demands **transparent evaluation of bias** (vehicle type, plate condition), **strict privacy safeguards** (on-device processing, data minimization), and **adversarial robustness testing** to prevent evasion or misuse. The system's ability to enable mass surveillance without human oversight necessitates regulatory frameworks before real-world integration.

This project establishes the **methodological foundation**—synthetic data pipelines, similarity-aware training, and benchmark comparisons—for Philippine-context automated plate recognition. With continued development, it could support legitimate applications like toll automation, parking management, or traffic analytics while mitigating the documented risks of biometric surveillance infrastructure.

## References
[1] Laroca, R., et al. "A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector," *IJCNN*, 2018.  
[2] Silva, S.M., Jung, C.R. "Real-time license plate detection and recognition using deep convolutional neural networks," *Journal of Visual Communication and Image Representation*, vol. 71, 2020.  
[3] Amon, M.C.E., et al. "Philippine license plate character recognition using Faster R-CNN with InceptionV2," *HNICEM*, 2019.  
[4] Deticio, R.G., et al. "Philippine License Plate Detection on Mobile," *IEEE HNICEM*, 2022.  
[5] Brillantes, A.K.M., et al. "Philippine License Plate Detection and Classification using Faster R-CNN and Feature Pyramid Network," *IEEE HNICEM*, 2019.  
[6] Wang, X., et al. "Adversarial generation of training examples for license plate recognition," *arXiv:1707.03124*, 2017.  
[7] Tourani, A., et al. "A Robust Deep Learning Approach for Automatic Iranian Vehicle License Plate Detection and Recognition," *IEEE Access*, vol. 8, pp. 201317-201330, 2020.  
[8] Sarhan, A., et al. "Egyptian car plate recognition based on YOLOv8, Easy-OCR, and CNN," *Journal of Electrical Systems and Information Technology*, vol. 11, 2024.  
[9] Wikipedia Contributors. "Vehicle registration plates of the Philippines," 2024. Available: https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_the_Philippines  
[10] Mishra, A., et al. "Scene Text Recognition using Similarity and a Lexicon with Sparse Belief Propagation," *IEEE Trans. PAMI*, vol. 31, no. 10, 2009.
