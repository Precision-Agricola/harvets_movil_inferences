import os
import sys
import torch
import shutil
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

def download_dataset(api_key, workspace_name, project_name, dataset_version, dataset_format, dataset_dir):
    if os.path.exists(dataset_dir):
        print(f"El dataset ya existe en '{dataset_dir}'. No se descargará nuevamente.")
    else:
        print("Descargando el dataset desde Roboflow...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_name).project(project_name)
        version = project.version(dataset_version)
        version.download(dataset_format, location=dataset_dir)
        print(f"Dataset descargado en '{dataset_dir}'.")
    
    # Ajustar las rutas en data.yaml
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    # Obtener rutas absolutas
    dataset_abs_path = os.path.abspath(dataset_dir)
    data_yaml['train'] = os.path.join(dataset_abs_path, 'train', 'images').replace('\\', '/')
    data_yaml['val'] = os.path.join(dataset_abs_path, 'valid', 'images').replace('\\', '/')
    data_yaml['test'] = os.path.join(dataset_abs_path, 'test', 'images').replace('\\', '/')
    
    # Guardar los cambios
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    print("Archivo data.yaml actualizado con rutas absolutas.")

def train_model(data_yaml, model_output_dir, epochs=50, batch_size=16, imgsz=640):
    model_path = os.path.join(model_output_dir, 'best.pt')
    if os.path.exists(model_path):
        print(f"El modelo ya ha sido entrenado y se encuentra en '{model_path}'. No se reentrenará.")
    else:
        print("Entrenando el modelo con YOLOv8...")

        # Verificar si hay GPU disponible y registrar el dispositivo
        if torch.cuda.is_available():
            device = 'cuda'
            print("Entrenando en GPU.")
        else:
            device = 'cpu'
            print("Entrenando en CPU.")

        # Cargar el modelo y asignar el dispositivo
        model = YOLO('yolov8n.pt')

        # Especificar un nombre fijo para el entrenamiento
        run_name = 'custom_train'

        # Eliminar el directorio de entrenamiento previo si existe (opcional)
        run_dir = os.path.join('runs', 'detect', run_name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)

        # Iniciar el entrenamiento y pasar el dispositivo
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            name=run_name
        )

        # Mueve el modelo entrenado a la carpeta de salida
        os.makedirs(model_output_dir, exist_ok=True)
        src_model_path = os.path.join('runs', 'detect', run_name, 'weights', 'best.pt')
        if os.path.exists(src_model_path):
            shutil.copy(src_model_path, model_path)
            print(f"Modelo entrenado y guardado en '{model_path}'.")
        else:
            print(f"No se encontró el archivo '{src_model_path}'. Asegúrate de que el entrenamiento se completó correctamente.")

def main():
    # Configuración
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if api_key:
        print(f"RF API Key: {api_key[:4]}***")
    else:
        raise ValueError("No se pudo leer la API Key de Roboflow. Asegúrate de que está configurada en el archivo .env")

    workspace_name = 'psa01'
    project_name = 'merma_in_situ'
    dataset_version = 1
    dataset_format = 'yolov8'
    dataset_dir = os.path.join('datasets', 'merma_in_situ')
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    model_output_dir = os.path.join('models', 'merma_in_situ_v1')

    # Descarga del dataset
    download_dataset(api_key, workspace_name, project_name, dataset_version, dataset_format, dataset_dir)

    # Entrenamiento del modelo
    train_model(data_yaml, model_output_dir, epochs=50, batch_size=16, imgsz=640)

if __name__ == '__main__':
    main()

