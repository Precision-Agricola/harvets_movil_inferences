# Overview
This repository provides an Android app for object detection using YOLOv8, along with Python scripts for dataset synchronization, model training, and inference. The model works for object detection, not semantic segmentation, and supports inferences on both images and videos.

This repository leverages YOLO and SAM models to create datasets, synchronize labeled data, train and test models, which are then used in a mobile app for on-device inference.

### Requirements
The project can be run on CPU, GPU, and MPS. It's recommended to install PyTorch with GPU support, depending on your hardware. Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for instructions.

To sync datasets, create a `.env` file at the root directory and add:
    plaintext
    ROBOFLOW_API_KEY=<your_api_key>
Otherwise, contact the administrator for access.

# Project Structure
    plaintext
    C:.
    ├───android_app
    │   └───android_app
    │       ├───app
    │       │   └───src
    │       │       ├───androidTest
    │       │       │   └───java/com/surendramaran/yolov8tflite
    │       │       ├───main
    │       │       │   ├───assets
    │       │       │   ├───java/com/surendramaran/yolov8tflite
    │       │       │   └───res
    │       │       └───test
    │       │           └───java/com/surendramaran/yolov8tflite
    │       └───gradle/wrapper
    ├───config
    ├───runs/detect
    ├───scripts/data_preparation
    └───src/data_processing

# Main Commands
### Dataset Synchronization
To synchronize datasets from Roboflow:
    sh
    python scripts/sync_dataset.py
The script takes `datasets_sync.yaml` with the following structure:
    yaml
    datasets:
      - workspace: "psa00"
        project: "merma-in-situ"
        version: 1
        format: "yolov8"
        output_dir: "datasets/merma_in_situ"
Modify or create a new YAML for custom datasets.

### Model Training
Train a YOLOv8 model:
    sh
    python scripts/train_model.py
The training configuration should be specified in `training_config.yaml`:
    yaml
    model:
      name: "crop_segmentation"
      data_yaml: "datasets/crop_segmentation/data.yaml"
      output_dir: "models/crop_segmentation"

    training:
      epochs: 250
      batch_size: 16
      imgsz: 640
      device: "auto"

### Inference
Run inference on images or videos:
    sh
    python scripts/inference.py
Configuration example in `inference.yaml`:
    yaml
    model:
      path: "models/pineaple_fruit_count/best.pt"

    inference:
      imgsz: 640
      conf: 0.05

    paths:
      input: "data/pineaple/counting_data/pineaple_count.mp4"
      output: "data/pineaple/counting_data/output/pineaple_count"

### Export to TFLite
To use the trained model in the Android app (written in Kotlin), you need to convert the `.pt` model to `.tflite` format:
    sh
    python scripts/export_tflite.py
The exported `.tflite` model should then be added to the Android app's assets directory for on-device inference.
