import cv2
import os
import cv2
from pathlib import Path

def save_first_frames(input_root, output_folder):
    input_root = Path(input_root)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for folder in input_root.iterdir():
        if folder.is_dir():
            for mp4_file in folder.glob("*.mp4"):
                cap = cv2.VideoCapture(str(mp4_file))
                success, frame = cap.read()
                if success:
                    # Construct unique filename
                    output_filename = f"{folder.name}_{mp4_file.stem}.jpg"
                    output_path = output_folder / output_filename
                    cv2.imwrite(str(output_path), frame)
                    print(f"Saved: {output_path}")
                else:
                    print(f"Failed to read: {mp4_file}")
                cap.release()

video_dir = "/home/saumyas/franka/frankapy/data/franka_exp_data/"
output_dir = "/home/saumyas/franka/frankapy/data/franka_exp_data/init_frames"

save_first_frames(video_dir, output_dir)