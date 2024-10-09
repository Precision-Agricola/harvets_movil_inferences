import math
from pathlib import Path
from PIL import Image
from wasabi import Printer

def create_tiles(image_path: Path, output_dir: Path, tile_size: int = 512, overlap: float = 0.2):
    """
    Divide una imagen en mosaicos con un traslape especificado.

    Args:
        image_path (Path): Ruta a la imagen de entrada.
        output_dir (Path): Directorio de salida para los mosaicos.
        tile_size (int): Tamaño de los mosaicos en píxeles.
        overlap (float): Proporción de traslape entre mosaicos (0 a 1).
    """
    msg = Printer()
    if not image_path.exists():
        msg.fail(f"No se encontró la imagen: {image_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    width, height = image.size

    step = int(tile_size * (1 - overlap))
    cols = math.ceil((width - tile_size) / step) + 1
    rows = math.ceil((height - tile_size) / step) + 1

    tile_count = 0
    for row in range(rows):
        for col in range(cols):
            left = col * step
            upper = row * step
            right = min(left + tile_size, width)
            lower = min(upper + tile_size, height)

            if right <= left or lower <= upper:
                continue

            tile = image.crop((left, upper, right, lower))
            tile_name = f"tile_{row}_{col}.jpg"
            tile_path = output_dir / tile_name
            tile.save(tile_path)
            tile_count += 1

    msg.good(f"Se crearon {tile_count} mosaicos en {output_dir}")

