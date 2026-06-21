# TTPLA Mask R-CNN Torchvision

Implementation of Mask R-CNN using Torchvision for object detection and instance segmentation on the TTPLA dataset. This project focuses on detecting transmission-related objects such as cable and tower types from image data.

## Project Overview

This project builds a deep learning pipeline for detecting and segmenting power transmission line objects. The pipeline uses three main notebooks:

1. Trainer notebook for model training.
2. Evaluator notebook for COCO metric evaluation.
3. Tester notebook for inference and prediction export.

The model uses Mask R-CNN with a ResNet-50 FPN backbone. The model is trained on the TTPLA dataset using COCO-style annotations.

## Objectives

The main objectives of this project are:

- Train a Mask R-CNN model on the TTPLA dataset.
- Detect cable and tower objects from transmission line images.
- Generate bounding box and segmentation mask predictions.
- Evaluate model performance using COCO metrics.
- Export testing results into JSON, CSV, and image visualizations.
- Provide a reproducible baseline for further improvement.

## Dataset

The project uses the TTPLA dataset with COCO annotation format.

Dataset split:

| Split | Total Images |
|---|---:|
| Training | 842 |
| Testing | 400 |

Object classes:

| Model Label | Class Name |
|---:|---|
| 1 | cable |
| 2 | tower_lattice |
| 3 | tower_tucohy |
| 4 | tower_wooden |

The model uses 5 classes in total because Mask R-CNN includes one background class.

```text
4 object classes + 1 background class = 5 classes
```

## Model Architecture

The model uses:

```text
Mask R-CNN ResNet-50 FPN
```

Main components:

- ResNet-50 as feature extractor.
- Feature Pyramid Network for multi-scale object detection.
- Fast R-CNN head for bounding box prediction.
- Mask R-CNN head for instance segmentation.
- COCO pretrained weights as the initial model weight.

The classifier and mask heads are replaced to match the TTPLA class structure.

## Repository Structure

```text
ttpla-mask-rcnn-torchvision/
├── README.md
├── .gitignore
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   ├── 01_trainer_mask_rcnn_ttpla.ipynb
│   ├── 02_evaluator_mask_rcnn_ttpla.ipynb
│   └── 03_tester_mask_rcnn_ttpla.ipynb
│
├── configs/
│   └── config.yaml
│
├── docs/
│   └── experiment_summary.md
│
├── outputs/
│   ├── evaluation_metrics.json
│   ├── tester_summary.json
│   ├── loss_curve.png
│   ├── map_curve.png
│   └── sample_prediction.png
│
└── results/
    ├── predictions.csv
    └── class_prediction_stats.csv
```

## Training Configuration

Main training configuration:

| Parameter | Value |
|---|---:|
| Batch size | 4 |
| Epochs | 15 |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Evaluation interval | Every epoch |
| Score threshold | 0.50 |
| Mask threshold | 0.50 |

The model was trained using CUDA GPU.

## Training Process

The training process follows this workflow:

```text
Load TTPLA dataset
↓
Read COCO annotations
↓
Prepare PyTorch Dataset and DataLoader
↓
Load Mask R-CNN ResNet-50 FPN
↓
Replace classifier and mask heads
↓
Train model for 15 epochs
↓
Evaluate model after each epoch
↓
Save best checkpoint
↓
Export training metrics and curves
```

The training loss decreased during training, which indicates that the model learned from the dataset.

Final training summary:

| Metric | Value |
|---|---:|
| Final train loss | 0.7378 |
| Best epoch | 15 |
| Final BBox AP | 0.1915 |
| Final Segmentation AP | 0.1235 |

## Evaluation Results

The evaluator notebook uses COCO metrics to measure bounding box and segmentation performance.

Evaluation setting:

| Setting | Value |
|---|---:|
| Test images | 400 |
| Checkpoint | best_model.pth |
| Checkpoint epoch | 15 |
| Score threshold | 0.05 |
| Mask threshold | 0.50 |
| BBox predictions | 17,956 |
| Segmentation predictions | 17,956 |

Main evaluation results:

| Metric | BBox | Segmentation |
|---|---:|---:|
| AP | 0.2146 | 0.1214 |
| AP50 | 0.3602 | 0.2195 |
| AP75 | 0.2342 | 0.1254 |
| AP Small | 0.1022 | 0.0008 |
| AP Medium | 0.2849 | 0.0625 |
| AP Large | 0.2258 | 0.2416 |
| AR 1 | 0.1706 | 0.1503 |
| AR 10 | 0.3337 | 0.2106 |
| AR 100 | 0.3594 | 0.2148 |

## Testing Results

The tester notebook runs inference on all test images and exports predictions.

Testing setting:

| Setting | Value |
|---|---:|
| Test images processed | 400 |
| Score threshold | 0.30 |
| Mask threshold | 0.50 |
| Total predictions | 6,064 |
| Average predictions per image | 15.16 |
| Testing time | 2,867.63 seconds |

Class prediction statistics:

| Class | Total Predictions | Mean Score | Max Score | Mean Area |
|---|---:|---:|---:|---:|
| cable | 5,815 | 0.6793 | 0.9992 | 40,277.92 |
| tower_lattice | 249 | 0.6073 | 0.9673 | 76,941.33 |

The model mostly predicts the cable class. The tower_lattice class appears in smaller numbers. The tower_tucohy and tower_wooden classes do not appear in the testing result at a score threshold of 0.30.

## Output Files

Training output:

```text
best_model.pth
last_model.pth
metrics.json
final_evaluation.json
loss_curve.png
map_curve.png
sample_prediction.png
```

Evaluation output:

```text
evaluation_metrics.json
loss_curve_from_training_history.png
map_curve_from_training_history.png
sample_predictions_summary.json
sample prediction images
```

Testing output:

```text
predictions.json
predictions.csv
tester_summary.json
class_prediction_stats.csv
test prediction images
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in this order:

```text
1. notebooks/01_trainer_mask_rcnn_ttpla.ipynb
2. notebooks/02_evaluator_mask_rcnn_ttpla.ipynb
3. notebooks/03_tester_mask_rcnn_ttpla.ipynb
```

Recommended environment:

```text
Python 3.10+
CUDA GPU
Google Colab or local GPU environment
```

## Important Notes

Large files should not be uploaded directly to GitHub.

Do not upload:

```text
data/
dataset/
*.pth
*.pt
*.zip
runs/
checkpoints/
large output folders
```

Use Google Drive, Kaggle, Hugging Face, or GitHub Release for large files such as dataset and model checkpoint.

## Model Interpretation

The model already works as a baseline. It can detect and segment several transmission line objects. However, the performance is still limited.

Main findings:

- Bounding box performance is better than segmentation performance.
- Small object segmentation is very weak.
- The model predicts cable much more often than other classes.
- Some tower classes are not detected during testing at threshold 0.30.
- The model needs further improvement before production use.

## Improvement Plan

Recommended improvements:

- Add more training data for minority classes.
- Balance class distribution.
- Improve annotation quality for segmentation masks.
- Train for more epochs.
- Try larger backbones.
- Tune score threshold and mask threshold.
- Add stronger augmentation.
- Evaluate per-class AP.
- Compare with YOLO-based object detection models.
- Test the model on real field images.

## Conclusion

This project successfully implements a full Mask R-CNN pipeline for the TTPLA dataset. The pipeline covers data preparation, training, evaluation, testing, prediction export, and visualization. The final model is suitable as an initial baseline for transmission line object detection and instance segmentation, but it still requires improvement for more accurate and reliable results.
