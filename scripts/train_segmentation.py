"""
Script to train a yolo 11 segmentation model using ultralytics and yaml conf files

This scripts aims to train a model that create a multi-point segmentation mask that can
be used as polygon exportable to a geojson file. The model should be availabe to classify the crop type.

In further versions, the model should be able to identify growth stages of the crop.

"""
import sys
from pathlib import Path
import torch
import shutil
from ultralytics import YOLO
from wasabi import msg
import yaml
import argparse


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """
    Load the training configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config: dict):
    """
    Train the YOLOv11 segmentaion model based on the provided configuration.

    Args:
        config (dict): Training configuration.
    """
    model_name = config['model']['name']
    data_yaml = config['model']['data_yaml']
    model_output_dir = Path(config['model']['output_dir'])
    model_path = model_output_dir / 'best.pt'

    if model_path.exists():
        msg.info(f"Model already trained and exists at '{model_path}'. Skipping training.")
        return

    msg.info("Training YOLOv11 segmentation model...")

    # Determine the device: First try CUDA, then MPS, otherwise fallback to CPU
    if config['training']['device'] == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            msg.good("CUDA is available. Training on GPU.")
        elif torch.backends.mps.is_available():
            device = 'mps'
            msg.info("MPS is available. Training on MPS.")
        else:
            device = 'cpu'
            msg.warn("CUDA and MPS are not available. Training on CPU.")
    else:
        device = config['training']['device']
    
    msg.info(f"Training on {device.upper()}.")


    # Load the segmentation model /(weights) using ultralytics
    try:
        pretrained_model = 'yolov8n-seg.pt'
        model = YOLO(pretrained_model)
    except Exception as e:
        raise Exception(f"Failed to load {pretrained_model}: {e}")

    # Specify a fixed name for the training run
    run_name = f'{model_name}_train'

    # Remove previous training directory if it exists
    run_dir = Path('runs') / 'segment' / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    # Start the training
    model.train(
        data=data_yaml,
        epochs=config['training']['epochs'],
        batch=config['training']['batch_size'],
        imgsz=config['training']['imgsz'],
        device=device,  # Here we pass the dynamically selected device
        name=run_name
    )

    # Move the trained model to the output directory
    model_output_dir.mkdir(parents=True, exist_ok=True)
    src_model_path = run_dir / 'weights' / 'best.pt'
    if src_model_path.exists():
        shutil.copy(src_model_path, model_path)
        msg.good(f"Trained model saved to '{model_path}'.")
    else:
        msg.fail(f"Model file not found at '{src_model_path}'. Check if training completed successfully.")

def main(config_path: str = "config/training_config.yaml"):
    """
    Main function to load config and train the model.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    train_model(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to train a YOLOv8 model using Ultralytics and YAML config files.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to the training configuration YAML file.'
    )
    args = parser.parse_args()
    main(args.config)


