import argparse
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import imageio
import subprocess
from concurrent.futures import ProcessPoolExecutor

def load_and_resize_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img.resize((w // 2, h // 2))

def tile_images(images, tile_shape):
    rows, cols = tile_shape
    h, w = images[0].size[1], images[0].size[0]

    # Create a new image with extra space for labels
    label_padding_x = 80  # left margin for row labels
    label_padding_y = 80   # top margin for column labels
    total_width = cols * w + label_padding_x
    total_height = rows * h + label_padding_y

    # tiled = Image.new("RGB", (total_width, total_height), color="black")
    # draw = ImageDraw.Draw(tiled)
    tiled = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))  # transparent background
    # paste frames first so labels draw on top
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        tiled.paste(img, (label_padding_x + col * w,
                          label_padding_y + row * h))
    draw = ImageDraw.Draw(tiled)

    # Load font (fallback to default if Arial not found)
    font = ImageFont.truetype("DejaVuSans.ttf", 40)

    row_labels = ["Cam 0", "Cam 1", "Cam 2"]
    col_labels = ["\"blue cube\"", "\"Franka robot arm\"", "\"green cube\"", "\"red cube\""]

    # Draw column labels (top)
    for col, label in enumerate(col_labels):
        text_bbox = font.getbbox(label)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = label_padding_x + col * w + w // 2 - text_w // 2
        text_y = label_padding_y // 2 - text_h // 2
        draw.text((text_x, text_y), label, fill="white", font=font)

    # Row labels (left, vertically rotated)
    for row, label in enumerate(row_labels):
        bbox = font.getbbox(label)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Create a blank image to rotate
        # label_img = Image.new("RGB", (text_w + 10, text_h + 10), "black")
        label_img = Image.new("RGBA", (text_w + 10, text_h + 10), (0,0,0,0))

        label_draw = ImageDraw.Draw(label_img)
        label_draw.text((0, 0), label, fill="white", font=font)

        # Rotate and paste
        rotated = label_img.rotate(90, expand=True)
        # rotated = label_img

        # Adjust X so label is flush with the left side, with a small inset (e.g. 10px)
        x = 10  # Small padding from left edge
        y = label_padding_y + row * h + h // 2 - rotated.height // 2
        tiled.paste(rotated, (x, y), rotated)

    # # Paste images
    # for idx, img in enumerate(images):
    #     row, col = divmod(idx, cols)
    #     x = label_padding_x + col * w
    #     y = label_padding_y + row * h
    #     tiled.paste(img, (x, y))

    return tiled.convert("RGB")

def process_frame(i, input_dirs, output_dir, tile_shape):
    frames = []
    for dir in input_dirs:
        frame_path = os.path.join(dir, f"{i:04d}_overlay.png")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame {i:04d}_overlay.png not found in {dir}")
        frames.append(load_and_resize_image(frame_path))
    tiled_image = tile_images(frames, tile_shape)
    tiled_image.save(os.path.join(output_dir, f"{i:04d}.png"))

def main():
    parser = argparse.ArgumentParser(description="Process some directories and tile frames.")
    parser.add_argument('--input_dirs', nargs='+', required=True, help='List of input directories containing frames.')
    parser.add_argument('--output_dir', required=True, help='Output directory to save tiled frames.')
    parser.add_argument('--tile_shape', type=int, nargs=2, required=True, help='Shape of the tile (rows cols).')
    parser.add_argument('--frame_rate', type=int, default=10, help='Frame rate for the output video.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    assert len(args.input_dirs) == args.tile_shape[0] * args.tile_shape[1], \
        f"Expected {args.tile_shape[0] * args.tile_shape[1]} input dirs for tile shape {args.tile_shape}"

    n_frames = len(os.listdir(args.input_dirs[0]))
    assert n_frames % 2 == 0
    n_frames //= 2
    print(f"Processing {n_frames} frames...")

    with ProcessPoolExecutor() as executor:
        # process_frame(0, args.input_dirs, args.output_dir, args.tile_shape)
        futures = [
            executor.submit(process_frame, i, args.input_dirs, args.output_dir, args.tile_shape)
            for i in range(n_frames)
        ]
        for f in tqdm(futures):
            f.result()  # raises any exceptions

if __name__ == "__main__":
    main()