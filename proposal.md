# CSC173 Deep Computer Vision Project Proposal
**Student:** Gio Kiefer A. Sanchez, 2022-0025  
**Date:** 12/11/2025

## 1. Project Title 
Segmentation-Based Philippine License Plate Character Recognition from Procedurally Generated Synthetic Data

## 2. Problem Statement
Standard Optical Character Recognition (OCR) models and off-the-shelf text detectors are not designed around license plate fonts and layouts, and they typically require heavy preprocessing to cope with blur, noise, and difficult lighting in unconstrained road scenes. [1][2] These limitations are amplified in the Philippine context, where legacy and modern plate designs coexist, diverging from the datasets on which generic OCR systems are trained. [9] This project aims to build a Philippine-specific license plate character recognizer by training a segmentation-based model on synthetic images that replicate local plate styles, backgrounds, and capture artifacts — leveraging pretrained backbones or Ultralytics segmentation models where appropriate to accelerate convergence and improve sim-to-real performance. [4][7]

## 3. Objectives
- Design and implement a procedural data generation pipeline that synthesizes photorealistic Philippine license plate images (private, public, and motorcycle formats) with automatic pixel-level character mask annotations. [4][7]
- Build and/or fine-tune a semantic segmentation model (custom or pretrained) to detect and classify individual alphanumeric characters on license plates under varied imaging conditions; Ultralytics segmentation tooling may be leveraged for experiments and baselines. [5][10]
- Achieve robust recognition performance across diverse lighting, motion blur, emboss effects, and background clutter, targeting high character-level accuracy on held-out synthetic and real test sets. [1][6]
- Develop a post-processing module that leverages known Philippine plate formats to resolve ambiguous characters (e.g., 0 vs. O, 1 vs. I) and produce valid plate strings. [9][10]

## 4. Dataset Plan
- **Source:** Custom procedurally generated synthetic dataset for Philippine license plates, blended into varied real-world background images (streets, vehicles, pedestrians, buildings). [4][7]
- **Classes:** 36 character classes: A–Z and 0–9, each with corresponding binary segmentation masks. [5][10]
- **Acquisition / Generation Strategy:**
  - Collect a diverse pool of background images to model negative data (structural patterns and textures, organic textures, common objects in context, etc.).
  - Procedurally render plate templates for multiple Philippine formats (legacy designs, current standard plates, motorcycle plates), including graphic backgrounds and varying color schemes. [9]
  - Apply augmentations such as brightness and contrast changes, Gaussian and motion blur, embossing, perspective and affine transformations, additive noise, and compression artifacts to mimic real camera conditions. [2][4]
  - For each sample, generate binary segmentation masks for each character at the pixel level, apply synchronized transformations to both plate image and masks, and store the resulting mask maps as ground truth for semantic segmentation supervision. [7][10]
- **Size and Balance:**
  - Start with approximately 10,000–30,000 synthesized labeled images, with the ability to scale up as needed.
  - Explicitly balance character frequencies, ensuring uniform coverage of all letters and digits across generated plate strings. [4]

## 5. Technical Approach
- **Overall Pipeline:**
  1. Formalize Philippine plate patterns (e.g., LLL-DDDD, LLL-DD) and implement a string generator that samples valid plate sequences for different plate types. [9]
  2. Render plate strings onto template images (including variations with graphic backgrounds like the Rizal Monument) and generate per-character binary segmentation masks in plate coordinates.
  3. Composite plates onto random background images and apply a sequence of transformations (scaling, rotation, perspective, blur, illumination changes, embossing, noise) to approximate real capture conditions, while applying the same transforms to each character mask. [2][4]
  4. Store the augmented plate images paired with their corresponding multi-channel segmentation masks (one binary channel per character class) for supervised training. [7][10]

- **Semantic Segmentation Model and Training (pretrained/backbone options allowed):**
  - **Architecture Design:**
    - Implement an encoder–decoder (or fully convolutional) network that takes a license plate crop as input and outputs a segmentation map where each spatial location is assigned a character class (or background). The design may be implemented from scratch or built by fine-tuning a pretrained encoder/backbone (e.g., ResNet, EfficientNet, or Ultralytics/CSP-based backbones) to leverage learned low-level features and speed training. Using Ultralytics' segmentation models or similar off-the-shelf segmentation implementations is an option for baselines or production-ready experimentation. [5][10]
    - The encoder progressively downsamples features through stacked convolutional and pooling (or depthwise separable) layers; the decoder upsamples and refines predictions, optionally using skip connections to preserve fine spatial detail.
    - Use standard building blocks (Conv–BatchNorm–ReLU, max pooling, bilinear upsampling) configured to enable pixel-accurate character localization and classification. [1][8]
  - **Loss Function:**
    - Employ a pixel-wise cross-entropy loss to supervise the segmentation task, treating each spatial location as an independent classification problem across 37 classes (36 characters + background).
    - Optionally combine cross-entropy with auxiliary losses (e.g., Dice or focal loss) or apply class weighting to handle imbalanced character frequencies across the dataset.
  - **Implementation:**
    - Implement the model and training pipeline in a deep learning framework (e.g., PyTorch). Experiments may use Ultralytics (e.g., YOLOv8 segmentation tooling) or custom training loops depending on the experiment — pretrained weights and framework utilities are allowed to accelerate development and improve generalization. [5][8]
    - Use standard practices such as mini-batch training, learning rate scheduling, validation splits, and early stopping to avoid overfitting. [1][8]
    - Train primarily on synthetic data and validate on held-out synthetic and a small set of real Philippine plate images; consider fine-tuning on a small real labeled subset if needed. [4][7]

- **Post-processing:**
  - Apply connected-component analysis or morphological operations to the output segmentation map to extract individual character regions. [10]
  - For each detected character region, compute the dominant class label and spatial bounding box.
  - Sort detections from left to right and group them according to known plate layouts to reconstruct plate strings.
  - Apply rule-based corrections using Philippine format constraints (e.g., letter-only prefix positions, digit-only suffix) to resolve ambiguous or low-confidence predictions (0 vs. O, 1 vs. I, 8 vs. B). [9]
  - Optionally output multiple candidate plate strings with confidence scores for downstream verification or ranking. [6]

## 6. Expected Challenges & Mitigations
- **Challenge:** Sim-to-real generalization from synthetic training data to real road imagery, especially under extreme weather, occlusions, or unusual viewing angles. [4][6]
  - **Mitigation:** Use aggressive domain randomization in the generator (lighting, blur, camera pose, occlusions, plate wear) and validate on a curated set of real Philippine plate images, with the option to fine-tune the CNN on a small labeled real subset. [4][7]
- **Challenge:** Degradation from strong motion blur, low resolution, or heavy video compression, which can severely reduce character legibility and segmentation quality. [2]
  - **Mitigation:** Include blurred and low-resolution variants during synthetic generation, and explore simple enhancement steps (e.g., deblurring or super-resolution) before feeding images to the CNN in extreme cases.
- **Challenge:** Boundary artifacts and over-/under-segmentation at character edges, particularly when characters are adjacent or overlapping. [10]
  - **Mitigation:** Use higher-resolution feature maps and multi-scale prediction heads; apply morphological post-processing (dilation, erosion) to refine segmentation boundaries; augment training with edge-emphasized variants to encourage precise character boundary learning.
- **Challenge:** Handling the sim-to-real domain gap in segmentation masks, where synthetic masks may not perfectly capture real character boundaries due to font rendering, anti-aliasing, or wear on actual plates. [1][4]
  - **Mitigation:** Inject synthetic boundary noise and anti-aliasing artifacts during generation; use validation-based early stopping to prevent overfitting to perfect synthetic masks.

## References
[1] Laroca, R., Severo, E., Zanlorensi, L. A., Oliveira, L. S., Gonçalves, G. R., Schwartz, W. R., & Menotti, D. (2018, July). A robust real-time automatic license plate recognition based on the YOLO detector. In 2018 international joint conference on neural networks (ijcnn) (pp. 1-10). IEEE.  
[2] Azam, S., & Islam, M. M. (2016). Automatic license plate detection in hazardous condition. Journal of Visual Communication and Image Representation, 36, 172-186.  
[4] Björklund, T., Fiandrotti, A., Annarumma, M., Francini, G., & Magli, E. (2019). Robust license plate recognition using neural networks trained on synthetic images. Pattern Recognition, 93, 134-146.  
[5] Salemdeeb, M., & Ertürk, S. (2021). Full depth CNN classifier for handwritten and license plate characters recognition. PeerJ Computer Science, 7, e576.  
[6] Xu, Z., Yang, W., Meng, A., Lu, N., Huang, H., Ying, C., & Huang, L. (2018). Towards end-to-end license plate detection and recognition: A large dataset and baseline. In Proceedings of the European conference on computer vision (ECCV) (pp. 255-271).  
[7] Harrysson, O. (2019). License plate detection utilizing synthetic data from superimposition. Master Theses in Mathematical Sciences.  
[8] Wang, S. R., Shih, H. Y., Shen, Z. Y., & Tai, W. K. (2022). End-to-End High Accuracy License Plate Recognition Based on Depthwise Separable Convolution Networks. arXiv preprint arXiv:2202.10277.  
[9] Amon, M. C. E., Brillantes, A. K. M., Billones, C. D., Billones, R. K. C., Jose, J. A., Sybingco, E., ... & Bandala, A. (2019, November). Philippine license plate character recognition using faster R-CNN with InceptionV2. In 2019 IEEE 11th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management (HNICEM) (pp. 1-4). IEEE.  
[10] Naidu, U. G., Thiruvengatanadhan, R., Narayana, S., & Dhanalakshmi, P. (2022). Character level segmentation and recognition using CNN followed random forest classifier for NPR system. International Journal of Advanced Computer Science and Applications, 13(11).
