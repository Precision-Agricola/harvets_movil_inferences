import cv2
from pathlib import Path
from wasabi import Printer

def extract_frames(video_path: Path, output_dir: Path, frame_interval: int = 30):
    """
    Extrae frames de un video y los guarda en un directorio.

    Args:
        video_path (Path): Ruta al archivo de video.
        output_dir (Path): Directorio de salida para los frames.
        frame_interval (int): Intervalo de frames para guardar.
    """
    msg = Printer()
    if not video_path.exists():
        msg.fail(f"No se encontró el archivo de video: {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg.fail(f"No se pudo abrir el archivo de video: {video_path}")
        return

    frame_count = 0
    saved_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
    msg.good(f"Se guardaron {saved_frames} frames en {output_dir}")

