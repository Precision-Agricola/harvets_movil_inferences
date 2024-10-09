"""
In this code the inference is try to predict in images and video some of the models


    Arguments: 
        model(str) - path to the .pt file, most commont /trained_model_dataset.pt or tflite
        input(str) - path to the video or image file to run the inference, see the supported files
        output(str) - path for inference results


""" 
import sys
from pathlib import Path
import shutil
import yaml
import cv2
import argparse
from ultralytics import YOLO
from wasabi import msg

# Add the project root directory to Python's path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

def load_config(config_path: str = "config/inference_config.yaml") -> dict:
    """
    Load the inference configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_inference(model: YOLO, input_path: Path, output_path: Path, imgsz: int, conf: float):
    """
    Run inference on an image or video file.

    Args:
        model (YOLO): Loaded YOLO model.
        input_path (Path): Path to the input file.
        output_path (Path): Path to save the output.
        imgsz (int): Input image size.
        conf (float): Confidence threshold.
    """
    msg.info(f"Processing: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            msg.info(f"Processing image: {input_path}")
            results = model.predict(source=str(input_path), save=True, imgsz=imgsz, conf=conf)
            processed_path = Path(results[0].save_dir) / input_path.name
            output_file = output_path / input_path.name
            shutil.move(str(processed_path), str(output_file))
            msg.good(f"Image processed and saved to: {output_file}")
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            msg.info(f"Processing video: {input_path}")
            results = model.predict(source=str(input_path), save=True, imgsz=imgsz, conf=conf)
            processed_path = Path(results[0].save_dir) / input_path.name
            output_file = output_path / input_path.name
            shutil.move(str(processed_path), str(output_file))
            msg.good(f"Video processed and saved to: {output_file}")
        else:
            msg.warn(f"Unsupported file type: {input_path}")
    elif input_path.is_dir():
        msg.info(f"Processing directory: {input_path}")
        for file in input_path.iterdir():
            run_inference(model, file, output_path, imgsz, conf)
    else:
        msg.fail(f"Input path does not exist: {input_path}")

def main(config_path: str = "inference_config.yaml"):
    """
    Main function to load config and run inference.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    
    model_path = Path(config['model']['path'])
    input_path = Path(config['paths']['input']) #TODO: Add support for images or directories of images
    output_path = Path(config['paths']['output'])
    imgsz = config['inference']['imgsz']
    conf = config['inference']['conf']

    if not model_path.exists():
        msg.fail(f"Model not found at: {model_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    msg.info(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    run_inference(model, input_path, output_path, imgsz, conf) #FIX: The output file is not saved correctly

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference with a YOLOv8 model on images or videos.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference_config.yaml',
        help='Path to the configuration file'
    )
    args = parser.parse_args()
    main(args.config)

