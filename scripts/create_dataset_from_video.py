"""
Script to read a video and save specific frames to a directory.

This script uses OpenCV to process a video file, saving images of frames at 
regular intervals in an output directory. The video input path and output 
directory are handled using `pathlib.Path` for efficient file management. 
Status and progress messages are displayed in the console using the `wasabi` 
library for colorful outputs.

Usage:
    The script saves a frame every certain number of frames (defined by 
    `frame_interval`, default is 30) in the specified output folder.
    The output folder is automatically created if it does not exist.

Structure:
    - The input video is processed frame by frame.
    - Selected frames are saved as `.jpg` files in the output directory.
    - Informative and progress messages are shown during execution.

Libraries:
    - OpenCV: To read and process the video.
    - pathlib: To manage file paths efficiently.
    - wasabi: To print colored status messages in the console.

Args:
    - video_path (Path): Path to the input video file.
    - output_dir (Path): Path to the directory where frames will be saved.
    - frame_interval (int, optional): Interval of frames to save images.

Example usage:
    python save_frames.py
"""


import cv2
from pathlib import Path
from wasabi import Printer

def save_frames(video_path: Path, output_dir: Path, frame_interval: int = 30) -> None:
    """
    Reads a video and saves frames at specified intervals.

    Args:
        video_path (Path): Path to the input video file.
        output_dir (Path): Directory where the frames will be saved.
        frame_interval (int): Interval of frames to save (default: 30).
    """
    msg = Printer()

    if not video_path.exists():
        msg.fail(f"Video file not found: {video_path}")
        return

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        msg.good(f"Created output directory: {output_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg.fail(f"Could not open video file: {video_path}")
        return

    frame_count = 0
    saved_frames = 0

    msg.info(f"Processing video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_file = output_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(output_file), frame)
            saved_frames += 1
            msg.good(f"Saved frame {frame_count} to {output_file}")

        frame_count += 1

    cap.release()
    msg.info(f"Total frames saved: {saved_frames}")

if __name__ == "__main__":
    video_path = Path("data/pineaple/counting_data/pineaple_fruit_count_demo.mp4")
    output_dir = Path("data/pineaple/counting_data/raw_dataset")
    save_frames(video_path, output_dir, frame_interval=30)
 
