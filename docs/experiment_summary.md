# Experiment Summary

## Project Title

TTPLA Object Detection and Instance Segmentation Using Mask R-CNN Torchvision

## 1. Background

Transmission line image analysis needs object detection and segmentation to identify important components such as cables and towers. Manual inspection can take time and may produce inconsistent records. This experiment builds a deep learning baseline using Mask R-CNN to detect and segment transmission-related objects from the TTPLA dataset.

## 2. Objective

The objective of this experiment is to train, evaluate, and test a Mask R-CNN model for object detection and instance segmentation on the TTPLA dataset.

The specific objectives are:

1. Train a Mask R-CNN model using COCO-style TTPLA annotations.
2. Detect four transmission-related classes.
3. Generate bounding box and segmentation mask predictions.
4. Evaluate the model using COCO metrics.
5. Export prediction results into JSON, CSV, and image visualization files.

## 3. Dataset

Dataset used: TTPLA dataset.

Dataset format: COCO annotation format.

Dataset split:

| Split | Total Images |
|---|---:|
| Training | 842 |
| Testing | 400 |

Classes:

| Label | Class Name |
|---:|---|
| 1 | cable |
| 2 | tower_lattice |
| 3 | tower_tucohy |
| 4 | tower_wooden |

The model uses 5 classes because one class is reserved for background.

## 4. Model

The experiment uses Mask R-CNN with ResNet-50 FPN from Torchvision.

Model configuration:

| Component | Description |
|---|---|
| Model | Mask R-CNN |
| Backbone | ResNet-50 FPN |
| Pretrained weight | COCO pretrained weight |
| Detection head | Fast R-CNN predictor |
| Segmentation head | Mask R-CNN predictor |
| Number of classes | 5 |

## 5. Training Setup

Main training configuration:

| Parameter | Value |
|---|---:|
| Batch size | 4 |
| Epochs | 15 |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Evaluation interval | 1 epoch |
| Score threshold | 0.50 |
| Mask threshold | 0.50 |

Augmentation used:

- Horizontal flip
- Vertical flip
- Random brightness and contrast
- RGB shift
- Blur

## 6. Training Process

The training process consists of:

1. Load dataset and COCO annotations.
2. Convert annotations into PyTorch target format.
3. Load the Mask R-CNN model.
4. Replace model heads based on TTPLA classes.
5. Train the model for 15 epochs.
6. Evaluate the model after every epoch.
7. Save the best model checkpoint.
8. Export metrics and visualizations.

Training result:

| Metric | Value |
|---|---:|
| Final train loss | 0.7378 |
| Best epoch | 15 |
| Final BBox AP | 0.1915 |
| Final Segmentation AP | 0.1235 |

The training loss decreased from the early epoch to the final epoch. This shows that the model learned from the training data.

## 7. Evaluation Setup

The evaluator notebook uses the best checkpoint from training.

Evaluation configuration:

| Setting | Value |
|---|---:|
| Test images | 400 |
| Checkpoint | best_model.pth |
| Checkpoint epoch | 15 |
| Score threshold | 0.05 |
| Mask threshold | 0.50 |
| BBox predictions | 17,956 |
| Segmentation predictions | 17,956 |
| Evaluation time | 2,204.94 seconds |

## 8. Evaluation Result

COCO evaluation result:

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

## 9. Testing Setup

The tester notebook runs inference on all test images.

Testing configuration:

| Setting | Value |
|---|---:|
| Test images processed | 400 |
| Score threshold | 0.30 |
| Mask threshold | 0.50 |
| Total predictions | 6,064 |
| Average predictions per image | 15.16 |
| Testing time | 2,867.63 seconds |

## 10. Testing Result

Class prediction statistics:

| Class | Total Predictions | Mean Score | Max Score | Mean Area |
|---|---:|---:|---:|---:|
| cable | 5,815 | 0.6793 | 0.9992 | 40,277.92 |
| tower_lattice | 249 | 0.6073 | 0.9673 | 76,941.33 |

The tester result shows that cable is the dominant predicted class. The tower_lattice class appears in smaller numbers. The tower_tucohy and tower_wooden classes do not appear in the testing result at a score threshold of 0.30.

## 11. Interpretation

The model already works as an initial baseline. It can detect and segment objects from the TTPLA dataset. However, the model still has several limitations.

Key findings:

1. Bounding box performance is higher than segmentation performance.
2. Segmentation AP is still low.
3. Small object segmentation is very weak.
4. The model mostly predicts cable.
5. Some tower classes are not detected during testing.
6. The model is not yet ready for production use.

## 12. Output Files

Training outputs:

- best_model.pth
- last_model.pth
- metrics.json
- final_evaluation.json
- loss_curve.png
- map_curve.png
- sample_prediction.png

Evaluation outputs:

- evaluation_metrics.json
- sample_predictions_summary.json
- loss curve
- mAP curve
- sample prediction images

Testing outputs:

- predictions.json
- predictions.csv
- tester_summary.json
- class_prediction_stats.csv
- test prediction images

## 13. Limitations

This experiment has several limitations:

1. The model was trained only for 15 epochs.
2. Some classes may have fewer samples than others.
3. Small objects are difficult to detect and segment.
4. The model produces many cable predictions.
5. Segmentation mask quality still needs improvement.
6. The current threshold setting may reduce predictions for weaker classes.

## 14. Future Work

Recommended future work:

1. Add more annotated data for minority classes.
2. Balance the dataset across all classes.
3. Improve segmentation mask annotation quality.
4. Train the model for more epochs.
5. Tune learning rate, threshold, and augmentation strategy.
6. Evaluate per-class AP.
7. Try a stronger backbone.
8. Compare Mask R-CNN with YOLO-based models.
9. Test the model on real field images.
10. Deploy the model as a simple inference application.

## 15. Conclusion

The experiment successfully builds a complete Mask R-CNN pipeline for the TTPLA dataset. The pipeline includes training, evaluation, testing, prediction export, and visualization. The result shows that the model can detect several transmission-related objects, but its segmentation performance still needs improvement. This project is suitable as a baseline for further research and model optimization.
