""" This script is used to download the datasets in yolo format using the Roboflow API.
    
    The datasets are downloaded to datasets/.. directory.
    The .env should contain the ROBOFLOW_API_KEY.
"""

from pathlib import Path
from roboflow import Roboflow
from roboflow.adapters.rfapi import RoboflowError
from wasabi import msg
import yaml

# Add this file to the path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from config import ROBOFLOW_API_KEY

def load_config(config_path: str = "/config/datasets_sync.yaml") -> dict:
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration data.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        msg.fail(f"Configuration file should be placed at 'config/datasets_sync.yaml'.")
        raise SystemExit

def sync_dataset(rf: Roboflow, dataset_config: dict) -> None:
    """
    Synchronize a single dataset from Roboflow.

    Args:
        rf (Roboflow): Roboflow instance.
        dataset_config (dict): Configuration for the dataset.

    """
    workspace_name = dataset_config['workspace']
    project_name = dataset_config['project']
    dataset_version = dataset_config['version']
    dataset_format = dataset_config['format']
    dataset_dir = dataset_config['output_dir']

    dataset_path = Path(dataset_dir)
    
    #TODO: Validate that labels and images are present in the dataset
    if dataset_path.exists():
        msg.info(f"Dataset already exists in '{dataset_path}'. Skipping download.")
    else:
        msg.info(f"Downloading dataset {project_name} from {workspace_name}...")
        try:
            project = rf.workspace(workspace_name).project(project_name)
            version = project.version(dataset_version)
            version.download(dataset_format, location=str(dataset_path))
            msg.good(f"Dataset downloaded to '{dataset_path}'.")
        except RoboflowError as e:
            msg.fail(f"Error accessing Roboflow for {project_name}: {e}")
            return

    update_data_yaml(dataset_path)

def update_data_yaml(dataset_path: Path) -> None:
    """
    Update data.yaml file with absolute paths.

    Args:
        dataset_path (Path): Path to the dataset directory.
    """
    data_yaml_path = dataset_path / 'data.yaml'
    if not data_yaml_path.exists():
        msg.warn(f"data.yaml not found at {data_yaml_path}. Skipping update.")
        return

    with data_yaml_path.open('r') as f:
        data_yaml = yaml.safe_load(f)
    
    dataset_abs_path = dataset_path.resolve()
    data_yaml['train'] = str(dataset_abs_path / 'train' / 'images')
    data_yaml['val'] = str(dataset_abs_path / 'valid' / 'images')
    data_yaml['test'] = str(dataset_abs_path / 'test' / 'images')
    
    with data_yaml_path.open('w') as f:
        yaml.dump(data_yaml, f)
    
    msg.good(f"data.yaml updated with absolute paths for {dataset_path.name}.")

def main(config_path: str = "config/datasets_sync.yaml"):
    """
    Main function to synchronize all datasets specified in the config file.

    Args:
        config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)
    api_key = ROBOFLOW_API_KEY
    rf = Roboflow(api_key=api_key)

    for dataset in config['datasets']:
        sync_dataset(rf, dataset)

if __name__ == '__main__':
    main()
