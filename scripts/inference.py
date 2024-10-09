"""
Este script realiza la inferencia utilizando un modelo YOLOv8 en imágenes o videos y guarda los resultados.

Argumentos:
    model (str): Ruta al archivo .pt del modelo entrenado.
    input (str): Ruta al archivo de imagen o video para la inferencia.
    output (str): Ruta donde se guardarán los resultados de la inferencia.
"""
import sys
from pathlib import Path
import yaml
import argparse
from ultralytics import YOLO
from wasabi import msg

def load_config(config_path: str = "config/inference_config.yaml") -> dict:
    """
    Carga la configuración de inferencia desde un archivo YAML.

    Args:
        config_path (str): Ruta al archivo de configuración.

    Returns:
        dict: Datos de configuración.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_inference(model: YOLO, input_path: Path, output_path: Path, imgsz: int, conf: float):
    """
    Ejecuta la inferencia en una imagen o archivo de video.

    Args:
        model (YOLO): Modelo YOLO cargado.
        input_path (Path): Ruta al archivo de entrada o directorio.
        output_path (Path): Ruta donde se guardará la salida.
        imgsz (int): Tamaño de imagen de entrada.
        conf (float): Umbral de confianza.
    """
    msg.info(f"Procesando: {input_path}")

    if input_path.exists():
        # Ejecuta la predicción
        # TODO: agregar soporte para guardar en formato mp4 (optimización de video)
        model.predict(
            source=str(input_path),
            save=True,
            imgsz=imgsz,
            conf=conf,
            project=str(output_path.parent),
            name=output_path.name,
            exist_ok=True
        )
        msg.good(f"Inferencia completada. Resultados guardados en: {output_path}")
    else:
        msg.fail(f"La ruta de entrada no existe: {input_path}")

def main(config_path: str = "config/inference_config.yaml"):
    """
    Función principal para cargar la configuración y ejecutar la inferencia.

    Args:
        config_path (str): Ruta al archivo de configuración.
    """
    config = load_config(config_path)

    model_path = Path(config['model']['path'])
    input_path = Path(config['paths']['input'])
    output_path = Path(config['paths']['output'])
    imgsz = config['inference']['imgsz']
    conf = config['inference']['conf']

    if not model_path.exists():
        msg.fail(f"No se encontró el modelo en: {model_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    msg.info(f"Cargando el modelo desde: {model_path}")
    model = YOLO(str(model_path))

    run_inference(model, input_path, output_path, imgsz, conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ejecuta inferencia con un modelo YOLOv8 en imágenes o videos.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/inference_config.yaml',
        help='Ruta al archivo de configuración'
    )
    args = parser.parse_args()
    main(args.config)

