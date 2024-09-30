"""
In this code the inference is try to predict in images and video some of the models


    Arguments: 
        model(str) - path to the .pt file, most commont /trained_model_dataset.pt or tflite
        input(str) - path to the video or image file to run the inference, see the supported files
        output(str) - path for inference results


 #TODO - chain the absolute path for relative path (see the deafault conf file for the ultralytics app roam data
 #TODO -  
""" 
import os
import cv2
import argparse
from ultralytics import YOLO

def run_inference(model_path, input_path, output_path, imgsz=640, conf=0.5):
    # Cargar el modelo
    model = YOLO(model_path)

    # Verificar si la entrada es una imagen o un video
    if os.path.isfile(input_path):
        input_ext = os.path.splitext(input_path)[1].lower()
        if input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            is_image = True
        elif input_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_image = False
        else:
            raise ValueError("El archivo de entrada no es una imagen o un video soportado.")
    else:
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    if is_image:
        # Realizar inferencia en una imagen
        results = model.predict(source=input_path, save=True, imgsz=imgsz, conf=conf)
        # Mover la imagen procesada al directorio de salida
        processed_image_path = results[0].save_dir / results[0].path.name
        os.makedirs(output_path, exist_ok=True)
        output_image_path = os.path.join(output_path, os.path.basename(processed_image_path))
        os.rename(processed_image_path, output_image_path)
        print(f"Inferencia completada. Imagen guardada en: {output_image_path}")
    else:
        # Realizar inferencia en un video
        results = model.predict(source=input_path, save=True, imgsz=imgsz, conf=conf)
        # Mover el video procesado al directorio de salida
        processed_video_path = results[0].save_dir / results[0].path.name
        os.makedirs(output_path, exist_ok=True)
        output_video_path = os.path.join(output_path, os.path.basename(processed_video_path))
        os.rename(processed_video_path, output_video_path)
        print(f"Inferencia completada. Video guardado en: {output_video_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Realizar inferencia con un modelo YOLOv8 en una imagen o video.")
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo entrenado (.pt o .tflite)')
    parser.add_argument('--input', type=str, required=True, help='Ruta a la imagen o video de entrada')
    parser.add_argument('--output', type=str, default='outputs', help='Directorio de salida para los resultados')
    parser.add_argument('--imgsz', type=int, default=640, help='Tamaño de las imágenes de entrada')
    parser.add_argument('--conf', type=float, default=0.5, help='Umbral de confianza')
    return parser.parse_args()

def main():
    args = parse_args()
    run_inference(args.model, args.input, args.output, imgsz=args.imgsz, conf=args.conf)

if __name__ == '__main__':
    main()

