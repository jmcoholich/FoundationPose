import h5py
import numpy as np
import cv2
import os
import sys
import json
import pickle

thing = "plates"
ROOT_DIR = f"/data3/fp_data/stack_three_{thing}/demonstration_stack_three_{thing}_000"
INPUT_H5 = os.path.join(ROOT_DIR, "demo_000.h5")
OUTPUT_H5 = os.path.join(ROOT_DIR, "demo_000_annotations.h5")

CAM_NAMES = ["front_cam", "overhead_cam", "side_cam"]
OBJECT_LABELS = ["yellow plate", "orange plate", "teal plate"]
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


def format_box(box):
    # make sure the box is in format xmin, ymin, xmax, ymax
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

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
        object_annots = {}

        for obj_idx, obj_label in enumerate(OBJECT_LABELS):
            current_img = base_img.copy()
            box = None
            box_ready = False
            color = BOX_COLORS[obj_idx]

            cv2.setMouseCallback("Annotate", draw_box, param={'color': color})
            window_title = f"[{thing}] Frame {frame_idx}, Cam: {cam}, Obj: {obj_label}"
            cv2.setWindowTitle("Annotate", window_title)

            print(f"\nFrame {frame_idx}, Camera {cam}, Object: {obj_label}")
            print("Draw a box with mouse OR press:")
            print("  [o] → out of frame")
            print("  [x] → stop tracking")
            print("  [n] → skip this object (no label)")
            print("  [u] → undo drawn box")
            print("  [c] → cancel and exit")

            while True:
                cv2.imshow("Annotate", current_img)
                key = cv2.waitKey(0)

                if key == ord('u'):
                    current_img = base_img.copy()
                    box_ready = False
                    box = None
                    print("Undid bounding box. Draw again or choose a label.")
                    continue

                if box_ready:
                    object_annots[obj_label] = format_box(box)
                    print("Box saved.")
                    break
                elif key == ord('o'):
                    object_annots[obj_label] = "out_of_frame"
                    print("Marked out of frame.")
                    break
                elif key == ord('x'):
                    object_annots[obj_label] = "stop_tracking"
                    print("Marked stop tracking.")
                    break
                elif key == ord('n'):
                    print("Skipped this object — no label recorded.")
                    break
                elif key == ord('c'):
                    print("Cancelled. Exiting without saving.")
                    cv2.destroyAllWindows()
                    sys.exit(0)



        annotations[f"{frame_idx}_{cam}"] = object_annots

    # After all 3 cameras are annotated
    prompt = (
        f"[{thing}] Frame {frame_idx} complete.\n"
        "[s] save and continue, "
        "[1] +1 step, [5] +5, [0] +10, [9] +50, [c] cancel"
    )
    print("\n" + prompt)
    cv2.setWindowTitle("Annotate", prompt.replace("\n", " "))

    while True:
        key = cv2.waitKey(0)
        if key == ord('s'):
            frame_idx += 1
            break
        elif key in [ord('1'), ord('5'), ord('0'), ord('9')]:
            multiplier = {ord('1'): 1, ord('5'): 5, ord('0'): 10, ord('9'): 50}[key]
            frame_idx += multiplier
            print(f"Skipping ahead by {multiplier} timestep(s).")
            break
        elif key == ord('c'):
            print("Cancelled. Exiting without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)

cv2.destroyAllWindows()
# === Save Annotations ===
with open(OUTPUT_H5.replace(".h5", ".pkl"), 'wb') as f:
    pickle.dump(annotations, f)

print(f"\n✅ Annotations saved to: {OUTPUT_H5.replace('.h5', '.pkl')}")
