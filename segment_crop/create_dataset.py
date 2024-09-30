import math
import sys
from pathlib import Path
from PIL import Image
import wasabi

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError

def generate_submosaics(input_dir, output_dir, tile_size=8192, overlap=0.2):
    msg = wasabi.Printer()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    step = int(tile_size * (1 - overlap))

    image_files = list(input_path.glob('*.jpg'))
    if not image_files:
        msg.fail(f"No .jpg files found in {input_path}")
        sys.exit(1)

    for image_file in image_files:
        base_name = image_file.stem
        jpw_file = input_path / f"{base_name}.jpw"
        prj_file = input_path / f"{base_name}.prj"

        if not jpw_file.exists():
            msg.warn(f"World file {jpw_file.name} not found. Skipping {image_file.name}.")
            continue

        try:
            image = Image.open(image_file)
        except Exception as e:
            msg.warn(f"Cannot open image {image_file.name}: {e}")
            continue

        width, height = image.size

        try:
            with jpw_file.open('r') as f:
                world_file = f.readlines()
        except Exception as e:
            msg.warn(f"Cannot read world file {jpw_file.name}: {e}")
            continue

        if len(world_file) < 6:
            msg.warn(f"Incomplete world file {jpw_file.name}. Skipping {image_file.name}.")
            continue

        try:
            x_scale = float(world_file[0].strip())
            y_skew = float(world_file[1].strip())
            x_skew = float(world_file[2].strip())
            y_scale = float(world_file[3].strip())
            x_origin = float(world_file[4].strip())
            y_origin = float(world_file[5].strip())
        except Exception as e:
            msg.warn(f"Error parsing world file {jpw_file.name}: {e}")
            continue

        cols = int(math.ceil((width - tile_size) / step)) + 1
        rows = int(math.ceil((height - tile_size) / step)) + 1

        msg.info(f"Processing {image_file.name} into {rows * cols} tiles.")

        for row in range(rows):
            for col in range(cols):
                left = col * step
                upper = row * step
                right = left + tile_size
                lower = upper + tile_size

                if right > width:
                    right = width
                    left = width - tile_size
                if lower > height:
                    lower = height
                    upper = height - tile_size
                if left < 0 or upper < 0:
                    continue

                tile = image.crop((left, upper, right, lower))

                tile_name = f"{base_name}_Tile_{row}_{col}"
                tile_image_path = output_path / f"{tile_name}.jpg"

                try:
                    tile.save(tile_image_path, 'JPEG', quality=95)
                except Exception as e:
                    msg.warn(f"Cannot save tile {tile_image_path.name}: {e}")
                    continue

                new_x_origin = x_origin + (left * x_scale)
                new_y_origin = y_origin + (upper * y_scale)

                new_jpw_path = output_path / f"{tile_name}.jpw"
                try:
                    with new_jpw_path.open('w') as f:
                        f.write(f"{x_scale}\n")
                        f.write(f"{y_skew}\n")
                        f.write(f"{x_skew}\n")
                        f.write(f"{y_scale}\n")
                        f.write(f"{new_x_origin}\n")
                        f.write(f"{new_y_origin}\n")
                except Exception as e:
                        msg.warn(f"Cannot write world file {new_jpw_path.name}: {e}")
                        continue

                if prj_file.exists():
                    new_prj_path = output_path / f"{tile_name}.prj"
                    try:
                        new_prj_path.write_bytes(prj_file.read_bytes())
                    except Exception as e:
                        msg.warn(f"Cannot copy projection file to {new_prj_path.name}: {e}")
                        continue

        msg.good(f"Completed processing {image_file.name}.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate sub-mosaics from satellite images.')
    parser.add_argument('--input_dir', type=str, default=None, help='Path to the input directory.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory.')
    parser.add_argument('--tile_size', type=int, default=4000, help='Tile size in pixels (default: 8192).')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap fraction between tiles (default: 0.2).')

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    if args.input_dir is None:
        args.input_dir = Path(r'D:\crt_data\satelital_sample\agave_samples')
    if args.output_dir is None:
        args.output_dir = script_dir / 'output_tiles'

    generate_submosaics(args.input_dir, args.output_dir, args.tile_size, args.overlap)

