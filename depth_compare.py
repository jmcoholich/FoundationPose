"""Script to ensure depth and RGB frames are aligned (critical for FoundationPose)"""

import os
import pickle
import numpy as np
from PIL import Image, ImageOps

# Paths
pkl_demo_file = "demo_data/demo_000.pkl"
output_folder = "demo_data/stack_blocks_demo_000"

# Load the demo data
with open(pkl_demo_file, 'rb') as f:
    demo = pickle.load(f)

n_frames = len(demo['depth_frames'])

# Setup output directories
depth_dir = os.path.join(output_folder, 'depth')
os.makedirs(depth_dir, exist_ok=True)

rgb_out_dir = os.path.join(output_folder, 'rgb')
os.makedirs(rgb_out_dir, exist_ok=True)

overlay_dir = os.path.join(output_folder, 'depth_rgb_overlay')
os.makedirs(overlay_dir, exist_ok=True)

for i in range(n_frames):
    # --- Process the Depth Frame ---
    # Extract the cropped depth frame (adjust slicing as needed)
    depth_frame = demo['depth_frames'][i, 2, :, 140:500]

    # Normalize the depth data to the 0-255 range if it is not already.
    depth_array = np.array(depth_frame, dtype=np.float32)
    d_min, d_max = np.min(depth_array), np.max(depth_array)
    # Avoid division by zero in case the image is uniform.
    d_max = 1000
    if d_max - d_min > 0:
        depth_norm = ((depth_array - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = depth_array.astype(np.uint8)

    # Create a grayscale image from the normalized depth data.
    depth_img = Image.fromarray(depth_norm)
    # Save the normalized depth image
    depth_img.save(os.path.join(depth_dir, f'{i:04d}.png'))

    # --- Process the RGB Frame ---
    # Extract the cropped RGB frame (with channel reordering if needed)
    rgb_frame = demo['rgb_frames'][i, 2, :, 140:500, ::-1]
    rgb_img = Image.fromarray(rgb_frame.astype(np.uint8))
    # Save the original RGB image.
    rgb_img.save(os.path.join(rgb_out_dir, f'{i:04d}.png'))

    # --- Create and Save the Overlay ---
    # Convert the depth image to "L" mode (grayscale) if necessary.
    depth_gray = depth_img.convert("L")

    # Apply a color map to the grayscale depth image.
    # Here, "black" and "yellow" denote the endpoints of the colormap.
    depth_colored = ImageOps.colorize(depth_gray, black="blue", white="yellow")
    # save depth colorized
    depth_colored.save(os.path.join(depth_dir, f'{i:04d}_color.png'))

    # Ensure the rgb image is in RGB mode.
    rgb_img = rgb_img.convert("RGB")

    # Blend the RGB image with the colored depth image.
    # The alpha parameter controls how much of the depth image appears on top.
    overlay_img = Image.blend(rgb_img, depth_colored, alpha=0.5)

    # Save the overlay image.
    overlay_img.save(os.path.join(overlay_dir, f'{i:04d}.png'))
