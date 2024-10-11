"""
Script to export YOLOv8 models to TFLite format.

This script uses a YAML configuration file to specify export parameters
and utilizes the Ultralytics YOLO framework for model export.
"""

import os
import sys
from pathlib import Path
import shutil
import argparse
import yaml
from ultralytics import YOLO
from wasabi import msg

# Add the root directory to the Python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

def load_config(config_path: str) -> dict:
    """
    Load the export configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def export_model_to_tflite(config: dict):
    """
    Export the YOLOv8 model to TFLite format based on the provided configuration.

    Args:
        config (dict): Export configuration.
    """
    model_input_path = Path(config['model']['input_path'])
    export_dir = Path(config['model']['export_dir'])
    tflite_model_path = export_dir / 'best_saved_model' / 'best_float32.tflite'

    if tflite_model_path.exists():
        msg.info(f"Model already exported to TFLite at '{tflite_model_path}'. Skipping export.")
        return

    msg.info("Exporting model to TFLite format...")
    model = YOLO(model_input_path)
    model.export(format='tflite', imgsz=config['export']['imgsz'])

    # Check if the export was successful
    if tflite_model_path.exists():
        msg.good(f"Model exported to TFLite and saved at '{tflite_model_path}'.")
    else:
        msg.fail(f"File '{tflite_model_path}' not found. Ensure the export completed successfully.")
        
    # Optionally, you can also check for the float16 version
    tflite_model_path_float16 = export_dir / 'best_saved_model' / 'best_float16.tflite'
    if tflite_model_path_float16.exists():
        msg.good(f"Float16 model exported to TFLite and saved at '{tflite_model_path_float16}'.")
    else:
        msg.warn(f"Float16 model not found at '{tflite_model_path_float16}'. This may be expected depending on your export settings.")

def main(config_path: str):
    """
    Main function to load config and export the model.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    export_model_to_tflite(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to export YOLOv8 models to TFLite format.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/export_config.yaml',
        help='Path to the export configuration YAML file.'
    )
    args = parser.parse_args()
    main(args.config)

