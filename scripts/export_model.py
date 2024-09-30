import os
import shutil
from ultralytics import YOLO

def export_model_to_tflite(model_input_path, export_dir):
    tflite_model_path = os.path.join(export_dir, 'model.tflite')
    if os.path.exists(tflite_model_path):
        print(f"El modelo ya ha sido exportado a TFLite en '{tflite_model_path}'. No se volverá a exportar.")
    else:
        print("Exportando el modelo a formato TFLite...")
        model = YOLO(model_input_path)
        model.export(format='tflite', imgsz=640)
        # Mueve el modelo exportado a la carpeta de exportación
        os.makedirs(export_dir, exist_ok=True)
        src_tflite_path = os.path.join('runs', 'detect', 'custom_train', 'weights', 'best-fp16.tflite')
        if os.path.exists(src_tflite_path):
            shutil.move(src_tflite_path, tflite_model_path)
            print(f"Modelo exportado a TFLite y guardado en '{tflite_model_path}'.")
        else:
            print(f"No se encontró el archivo '{src_tflite_path}'. Asegúrate de que la exportación se completó correctamente.")

def main():
    # Configuración
    model_output_dir = os.path.join('models', 'merma_in_situ_v1')
    export_dir = model_output_dir
    model_input_path = os.path.join(model_output_dir, 'best.pt')

    # Exportación del modelo a TFLite
    export_model_to_tflite(model_input_path, export_dir)

if __name__ == '__main__':
    main()

