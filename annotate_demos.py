import h5py
import numpy as np
import cv2
import os
import sys

ROOT_DIR = "/data3/fp_data/stack_three_blocks/demonstration_stack_three_blocks_000"
INPUT_H5 = os.path.join(ROOT_DIR, "demo_000.h5")
OUTPUT_H5 = os.path.join(ROOT_DIR, "demo_000_annotations.h5")

CAM_NAMES = ["front_cam", "overhead_cam", "side_cam"]
OBJECT_LABELS = ["red block", "blue block", "green block"]
BOX_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # R, B, G

with h5py.File(INPUT_H5, 'r') as h5f:
    rgb_group = h5f['rgb_frames']
    T = rgb_group[CAM_NAMES[0]].shape[0]
    rgb_data = {cam: rgb_group[cam][:] for cam in CAM_NAMES}

annotations = {}

# === UI Globals ===
drawing = False
ix = iy = -1
current_img = None
box = None
box_ready = False

def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, box, box_ready, current_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = current_img.copy()
        cv2.rectangle(img_copy, (ix, iy), (x, y), param['color'], 2)
        cv2.imshow("Annotate", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box = (ix, iy, x, y)
        cv2.rectangle(current_img, (ix, iy), (x, y), param['color'], 2)
        cv2.imshow("Annotate", current_img)
        box_ready = True

cv2.namedWindow("Annotate")

frame_idx = 0
while frame_idx < T:
    for cam in CAM_NAMES:
        base_img = rgb_data[cam][frame_idx].copy()

        object_annots = []

        for obj_idx, obj_label in enumerate(OBJECT_LABELS):
            current_img = base_img.copy()
            box = None
            box_ready = False
            color = BOX_COLORS[obj_idx]

            cv2.setMouseCallback("Annotate", draw_box, param={'color': color})

            print(f"\nFrame {frame_idx}, Camera {cam}, Object: {obj_label}")
            print("Draw a box with mouse OR press:")
            print("  [o] → out of frame")
            print("  [x] → stop tracking")
            print("  [c] → cancel and exit")

            while True:
                cv2.imshow("Annotate", current_img)
                key = cv2.waitKey(0)

                if box_ready:
                    object_annots.append({"label": obj_label, "value": box})
                    print("Box saved.")
                    break
                elif key == ord('o'):
                    object_annots.append({"label": obj_label, "value": "out_of_frame"})
                    print("Marked out of frame.")
                    break
                elif key == ord('x'):
                    object_annots.append({"label": obj_label, "value": "stop_tracking"})
                    print("Marked stop tracking.")
                    break
                elif key == ord('c'):
                    print("Cancelled. Exiting without saving.")
                    cv2.destroyAllWindows()
                    sys.exit(0)

        annotations[f"{frame_idx}_{cam}"] = object_annots

        print("All 3 objects annotated for this view.")
        print("[s] save and continue, [1]/[5]/[0]/[9] = skip ahead, [c] cancel")
        while True:
            key = cv2.waitKey(0)
            if key == ord('s'):
                break
            elif key in [ord('1'), ord('5'), ord('0'), ord('9')]:
                multiplier = {ord('1'): 1, ord('5'): 5, ord('0'): 10, ord('9'): 50}[key]
                frame_idx += multiplier
                break
            elif key == ord('c'):
                print("Cancelled. Exiting without saving.")
                cv2.destroyAllWindows()
                sys.exit(0)

    frame_idx += 1

cv2.destroyAllWindows()

# === Save Annotations ===
with h5py.File(OUTPUT_H5, 'w') as f:
    import json
    dt = h5py.string_dtype(encoding='utf-8')
    group = f.create_group("object_annotations")
    for key, val in annotations.items():
        group.create_dataset(key, data=json.dumps(val), dtype=dt)

print(f"\n✅ Annotations saved to: {OUTPUT_H5}")
