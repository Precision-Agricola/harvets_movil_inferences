"""Inteface for extracting frames from a video file."""

import argparse
from pathlib import Path
from src.data_processing.video_utils import extract_frames

def main():
    parser = argparse.ArgumentParser(description="Extraer frames de un video.")
    parser.add_argument("--video_path", type=str, required=True, help="Ruta al archivo de video.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio de salida para los frames.")
    parser.add_argument("--frame_interval", type=int, default=30, help="Intervalo de frames para guardar.")

    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    frame_interval = args.frame_interval

    extract_frames(video_path, output_dir, frame_interval)

if __name__ == "__main__":
    main()

