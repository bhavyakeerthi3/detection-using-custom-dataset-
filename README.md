# YOLO11 Object Detection on Custom Dataset

## Overview

This project demonstrates training and deploying a custom object detection model using YOLO11 and a dataset hosted on Roboflow. The notebook covers the complete workflow from dataset acquisition to model deployment and inference.

### Features

- YOLO11 object detection
- Custom dataset training
- Roboflow dataset integration
- Model evaluation and validation
- Prediction on test images
- Deployment to Roboflow
- Inference using deployed models

---

## Project Workflow

```text
Custom Dataset (Roboflow)
           │
           ▼
 Dataset Download
           │
           ▼
    YOLO11 Training
           │
           ▼
   Model Evaluation
           │
           ▼
    Best Model Weights
           │
           ▼
 Test Image Prediction
           │
           ▼
 Model Deployment
           │
           ▼
 Real-Time Inference
```

---

## Technologies Used

- Python
- YOLO11 (Ultralytics)
- PyTorch
- Roboflow
- OpenCV
- Supervision
- Google Colab

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/yolo11-object-detection.git
cd yolo11-object-detection
```

Install dependencies:

```bash
pip install ultralytics supervision roboflow inference
```

---

## Dataset

The dataset is managed through Roboflow and downloaded in YOLO-compatible format.

Example:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("yolov11")
```

---

## Training

Train the YOLO11 model:

```bash
yolo task=detect mode=train \
model=yolo11s.pt \
data=data.yaml \
epochs=10 \
imgsz=640
```

### Training Configuration

| Parameter | Value |
|------------|--------|
| Model | YOLO11s |
| Epochs | 10 |
| Image Size | 640 |
| Task | Object Detection |

---

## Validation

Evaluate model performance:

```bash
yolo task=detect mode=val \
model=runs/detect/train/weights/best.pt \
data=data.yaml
```

Metrics generated:

- mAP@50
- mAP@50-95
- Precision
- Recall
- Confusion Matrix

---

## Prediction

Run inference on custom images:

```bash
yolo task=detect mode=predict \
model=runs/detect/train/weights/best.pt \
source=test/images \
conf=0.25 \
save=True
```

Results are stored in:

```text
runs/detect/predict/
```

---

## Visualization

Bounding boxes and labels are visualized using the Supervision library.

```python
import supervision as sv

detections = sv.Detections.from_ultralytics(result)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(image, detections)
annotated_image = label_annotator.annotate(
    annotated_image,
    detections
)
```

---

## Deployment

Deploy the trained model to Roboflow:

```python
project.version(dataset.version).deploy(
    model_type="yolov11",
    model_path="runs/detect/train/"
)
```

---

## Inference Using Deployed Model

```python
import inference

model = inference.get_model(model_id)

results = model.infer(
    image,
    confidence=0.4,
    overlap=30
)
```

---

## Output Structure

```text
runs/
└── detect/
    ├── train/
    │   ├── weights/
    │   │   ├── best.pt
    │   │   └── last.pt
    │   ├── results.png
    │   └── confusion_matrix.png
    │
    └── predict/
        └── prediction_images
```

---

## Results

The project generates:

- Trained YOLO11 model
- Best model weights
- Validation metrics
- Confusion matrix
- Prediction visualizations
- Deployment-ready model

---

## Future Improvements

- Hyperparameter tuning
- Data augmentation strategies
- Real-time webcam detection
- Edge deployment
- Model quantization
- Multi-class object detection expansion

---

## References

- Ultralytics YOLO Documentation
- Roboflow Documentation
- OpenCV Documentation

---


