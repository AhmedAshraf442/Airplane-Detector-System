# Airplane Detector System

## Overview
The **Airplane Detector System** is designed to detect airplanes in satellite imagery using deep learning models like YOLO and Faster R-CNN. The trained models are applied to real-world satellite images, including a test dataset featuring Cairo Airport, to generate accurate predictions with bounding boxes.

## Objectives
- Detect airplanes in satellite imagery using YOLO or Faster R-CNN.
- Train on the Airbus Tiles dataset and evaluate performance metrics.
- Apply the trained models to a satellite image of Cairo Airport.
- Visualize detections using bounding boxes.

## Dataset
- **Train Dataset:** [Airbus Tiles Dataset](https://universe.roboflow.com/rashad-pcyex/airbus_tiles/dataset/1)
- **Test Dataset:** [Cairo Airport Dataset](https://drive.google.com/file/d/18rKHi7fVXnO5pHVak0_0wKxgiZfsy5R_/view)

## Methodology
### 1. Read & Visualize Dataset
- Extract images from the dataset directory.
- Visualize original images with bounding boxes.

### 2. Preprocessing
- Apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance image contrast.
- Apply **Sharpening Filter** to improve edge detection.
- Save the final processed images for model training.

### 3. Model Training (YOLOv8)
- Train the YOLO model on the processed dataset.
- Performance results:
  - **Precision:** 0.96
  - **Recall:** 0.92
  - **mAP50:** 0.94
  - **mAP50-95:** 0.70
- Apply the trained model on the test set.

### 4. Application to Cairo Airport Image
- Perform inference on image slices.
- Draw bounding boxes on the full image.
- Save and visualize the output image with detected airplanes.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- Matplotlib, NumPy, and Pandas

### Installation
Clone the repository:
```bash
git clone https://github.com/your_username/Airplane-Detector-System.git
cd Airplane-Detector-System
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model
1. **Preprocess Dataset**
   ```bash
   python preprocess.py
   ```
2. **Train the YOLO Model**
   ```bash
   python train.py --epochs 50 --batch 16
   ```
3. **Run Inference on Cairo Airport Image**
   ```bash
   python inference.py --image path/to/cairo_airport_image.tif
   ```

## Results
- The model achieved high precision and recall on the Airbus dataset.
- The trained model successfully detected airplanes in satellite images, including the Cairo Airport test set.

## Author
Written by **Eng. Ahmed Ashraf**

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Airbus Tiles Dataset contributors.
- Open-source deep learning frameworks and libraries used in this project.

