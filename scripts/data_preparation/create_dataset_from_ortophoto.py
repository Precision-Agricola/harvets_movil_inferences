"""Interface for creating a dataset from an ortophoto image"""
import argparse
from pathlib import Path
from src.data_processing.ortophoto_utils import create_tiles

def main():
    parser = argparse.ArgumentParser(description="Crear mosaicos a partir de una imagen.")
    parser.add_argument("--image_path", type=str, required=True, help="Ruta a la imagen de entrada.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio de salida para los mosaicos.")
    parser.add_argument("--tile_size", type=int, default=512, help="Tamaño de los mosaicos en píxeles.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Proporción de traslape entre mosaicos (0 a 1).")

    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    tile_size = args.tile_size
    overlap = args.overlap

    create_tiles(image_path, output_dir, tile_size, overlap)

if __name__ == "__main__":
    main()

