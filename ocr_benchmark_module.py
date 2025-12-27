"""
OCR Model Benchmarking Module
Handles loading and inference for different OCR models with consistent interface
Supports both segmentation models and end-to-end OCR models
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class OCRModelWrapper:
    """Base wrapper class for OCR models"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.model_path = model_path
        # Character mapping for A-Z, 0-9 (36 classes)
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
    def preprocess(self, image):
        """Preprocess image - to be overridden"""
        raise NotImplementedError
        
    def predict(self, image):
        """Make prediction - to be overridden"""
        raise NotImplementedError


class CustomSegmentationOCR(OCRModelWrapper):
    """Wrapper for your custom segmentation model"""
    
    def __init__(self, model_path, conf_threshold=0.25, **kwargs):
        super().__init__(model_path, **kwargs)
        self.conf_threshold = conf_threshold
        
        # Load your segmentation model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
    def preprocess(self, image):
        """Preprocess image for your segmentation model"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Store original size for bbox conversion
        self.orig_width, self.orig_height = image.size
        
        # Your model's preprocessing (adjust as needed)
        # Most YOLO-style models expect specific input sizes
        img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        
        return img_tensor, image
    
    def predict(self, image):
        """Predict characters and their bounding boxes"""
        img_tensor, orig_image = self.preprocess(image)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # Parse detections and sort by x-coordinate (left to right)
        detections = self._parse_detections(output)
        
        # Sort by x-coordinate to get reading order
        detections = sorted(detections, key=lambda x: x['bbox'][0])
        
        # Extract text
        text = ''.join([det['char'] for det in detections])
        
        return text, detections
    
    def _parse_detections(self, output):
        """Parse model output to get character detections"""
        detections = []
        
        # This depends on your model's output format
        # Adjust based on how your segmentation model outputs predictions
        # For now, assuming output is [batch, num_detections, (x, y, w, h, conf, class_id)]
        
        if isinstance(output, (list, tuple)):
            output = output[0]  # Take first element if it's a list/tuple
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Filter by confidence
        for det in output:
            if len(det) >= 6:
                x, y, w, h, conf, class_id = det[:6]
                
                if conf > self.conf_threshold:
                    class_id = int(class_id.item() if torch.is_tensor(class_id) else class_id)
                    
                    if class_id < len(self.chars):
                        detections.append({
                            'char': self.chars[class_id],
                            'bbox': (float(x), float(y), float(w), float(h)),
                            'confidence': float(conf)
                        })
        
        return detections


class CRNNModel(OCRModelWrapper):
    """Wrapper for CRNN models (crnn_synth90k.pt and crnn.pth)"""
    
    def __init__(self, model_path, img_height=32, img_width=100, **kwargs):
        super().__init__(model_path, **kwargs)
        self.img_height = img_height
        self.img_width = img_width
        
        # CRNN character set (adjust based on actual model)
        # Most CRNN models use: digits + lowercase + uppercase
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.blank_label = len(self.alphabet)  # CTC blank token
        
        # Load model
        self._load_model()
        
        # Preprocessing - CRNN typically uses grayscale
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def _load_model(self):
        """Load CRNN model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, 
                                   weights_only=False)
            
            if isinstance(checkpoint, dict):
                # Check different possible keys
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need model architecture - using a basic CRNN
                    self.model = self._build_crnn()
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'net' in checkpoint:
                    self.model = checkpoint['net']
                else:
                    # Try to use the dict as state_dict
                    self.model = self._build_crnn()
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                # Assume it's the model directly
                self.model = checkpoint
                
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded CRNN model from {Path(self.model_path).name}")
            
        except Exception as e:
            print(f"✗ Error loading {Path(self.model_path).name}: {e}")
            raise
    
    def _build_crnn(self):
        """Build basic CRNN architecture"""
        class BidirectionalLSTM(nn.Module):
            def __init__(self, nIn, nHidden, nOut):
                super(BidirectionalLSTM, self).__init__()
                self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
                self.embedding = nn.Linear(nHidden * 2, nOut)

            def forward(self, x):
                recurrent, _ = self.rnn(x)
                T, b, h = recurrent.size()
                t_rec = recurrent.view(T * b, h)
                output = self.embedding(t_rec)
                output = output.view(T, b, -1)
                return output

        class CRNN(nn.Module):
            def __init__(self, imgH=32, nc=1, nclass=37, nh=256):
                super(CRNN, self).__init__()
                
                # CNN
                self.cnn = nn.Sequential(
                    nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),
                    nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)),
                    nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
                )
                
                # RNN
                self.rnn = nn.Sequential(
                    BidirectionalLSTM(512, nh, nh),
                    BidirectionalLSTM(nh, nh, nclass)
                )

            def forward(self, x):
                conv = self.cnn(x)
                b, c, h, w = conv.size()
                conv = conv.squeeze(2)  # [b, c, w]
                conv = conv.permute(2, 0, 1)  # [w, b, c]
                output = self.rnn(conv)
                return output
        
        return CRNN(nclass=len(self.alphabet) + 1)  # +1 for CTC blank
    
    def preprocess(self, image):
        """Preprocess image for CRNN"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
            
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, image):
        """Predict text from image"""
        img_tensor = self.preprocess(image)
        
        with torch.no_grad():
            preds = self.model(img_tensor)
            
        text = self._ctc_decode(preds)
        return text.upper()  # Convert to uppercase to match your format
    
    def _ctc_decode(self, preds):
        """Simple CTC greedy decoder"""
        # preds shape: [seq_len, batch, num_classes]
        _, preds = preds.max(2)  # [seq_len, batch]
        preds = preds.transpose(1, 0).contiguous().view(-1)  # [seq_len]
        
        # CTC decoding: remove duplicates and blanks
        prev_char = None
        result = []
        
        for char_idx in preds:
            char_idx = int(char_idx)
            
            if char_idx != self.blank_label and char_idx != prev_char:
                if char_idx < len(self.alphabet):
                    result.append(self.alphabet[char_idx])
            
            prev_char = char_idx
        
        return ''.join(result)


class DatasetHelper:
    """Helper class to load and parse dataset annotations"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file with character bboxes"""
        bboxes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    bboxes.append({
                        'class_id': class_id,
                        'char': self.chars[class_id] if class_id < len(self.chars) else '?',
                        'bbox': (x_center, y_center, width, height)
                    })
        
        # Sort by x_center to get reading order
        bboxes = sorted(bboxes, key=lambda x: x['bbox'][0])
        
        return bboxes
    
    def get_ground_truth(self, label_path):
        """Get ground truth text from label file"""
        bboxes = self.parse_yolo_label(label_path)
        return ''.join([b['char'] for b in bboxes]), bboxes
    
    def get_sample(self, split='test', idx=0):
        """Get a sample image and its annotations"""
        split_path = self.dataset_path / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        # Get all images
        images = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        
        if idx >= len(images):
            raise IndexError(f"Index {idx} out of range. Only {len(images)} images in {split} set.")
        
        img_path = images[idx]
        label_path = labels_path / f"{img_path.stem}.txt"
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Parse annotations
        ground_truth, bboxes = self.get_ground_truth(label_path)
        
        return {
            'image': image,
            'image_path': str(img_path),
            'ground_truth': ground_truth,
            'bboxes': bboxes,
            'label_path': str(label_path)
        }
    
    def visualize_sample(self, sample, predictions=None, figsize=(15, 5)):
        """Visualize a sample with ground truth and optional predictions"""
        fig, axes = plt.subplots(1, 2 if predictions else 1, figsize=figsize)
        
        if predictions is None:
            axes = [axes]
        
        # Ground truth visualization
        img = np.array(sample['image'])
        axes[0].imshow(img)
        axes[0].set_title(f"Ground Truth: {sample['ground_truth']}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Draw bounding boxes
        h, w = img.shape[:2]
        for bbox_info in sample['bboxes']:
            x_c, y_c, box_w, box_h = bbox_info['bbox']
            
            # Convert from normalized YOLO format to pixel coordinates
            x1 = (x_c - box_w/2) * w
            y1 = (y_c - box_h/2) * h
            box_w_px = box_w * w
            box_h_px = box_h * h
            
            rect = patches.Rectangle((x1, y1), box_w_px, box_h_px,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
            
            # Add character label
            axes[0].text(x1, y1-5, bbox_info['char'], 
                        color='lime', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Predictions visualization
        if predictions:
            axes[1].imshow(img)
            pred_text = predictions.get('text', 'N/A')
            correct = '✓' if pred_text == sample['ground_truth'] else '✗'
            axes[1].set_title(f"Prediction: {pred_text} {correct}", 
                            fontsize=12, fontweight='bold',
                            color='green' if correct == '✓' else 'red')
            axes[1].axis('off')
        
        plt.tight_layout()
        return fig


def load_model(model_name, models_dir='models'):
    """
    Load a model by name
    
    Args:
        model_name: One of 'custom_ocr', 'custom_ocr_refined', 'crnn_synth90k', 'crnn'
        models_dir: Base directory containing models
    
    Returns:
        OCRModelWrapper instance
    """
    models_dir = Path(models_dir)
    
    if model_name == 'custom_ocr':
        model_path = models_dir / 'custom_ocr'
        return CustomSegmentationOCR(model_path)
    
    elif model_name == 'custom_ocr_refined':
        model_path = models_dir / 'custom_ocr_refined'
        return CustomSegmentationOCR(model_path)
    
    elif model_name == 'crnn_synth90k':
        model_path = models_dir / 'other_models' / 'crnn_synth90k.pt'
        return CRNNModel(model_path)
    
    elif model_name == 'crnn':
        model_path = models_dir / 'other_models' / 'crnn.pth'
        return CRNNModel(model_path)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: custom_ocr, custom_ocr_refined, crnn_synth90k, crnn")


def benchmark_model(model, dataset_helper, split='test', num_samples=100):
    """
    Benchmark a model on dataset
    
    Args:
        model: OCRModelWrapper instance
        dataset_helper: DatasetHelper instance
        split: 'train', 'test', or 'valid'
        num_samples: Number of samples to test
    
    Returns:
        Dictionary with accuracy metrics
    """
    correct = 0
    total = 0
    
    for i in range(num_samples):
        try:
            sample = dataset_helper.get_sample(split, i)
            
            if isinstance(model, CustomSegmentationOCR):
                pred_text, _ = model.predict(sample['image'])
            else:
                pred_text = model.predict(sample['image'])
            
            if pred_text == sample['ground_truth']:
                correct += 1
            total += 1
            
        except IndexError:
            break
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }