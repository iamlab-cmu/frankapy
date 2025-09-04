import cv2
import os

def convert_mp4_to_pngs(video_path, output_dir, prefix="frame"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        output_path = os.path.join(output_dir, f"{prefix}_{frame_idx:05d}.png")
        cv2.imwrite(output_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames to {output_dir}")

# Example usage:
video_path = "/home/saumyas/franka/frankapy/data/franka_exp_data/2025_05_07_19_14_57ckpt_8oc8cvgn_run_succ/random_slide_pickup_0_goal_target_set0_success_None.mp4"
output_dir = "/home/saumyas/franka/frankapy/data/franka_exp_data/2025_05_07_19_14_57ckpt_8oc8cvgn_run_succ/"
convert_mp4_to_pngs(video_path, output_dir)
