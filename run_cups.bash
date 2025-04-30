#!/bin/bash
set -ex


INPUT_DIR="for_FoundationPose_cups"
PROMPTS=("orange cup" "white cup that is NOT teal" "teal cup")  # for lang-sam segmentations. Assumes same CAD model for all
CAM_NUMBER=1

# Determine if headless
HEADLESS_ARG=""
if [ -z "$DISPLAY" ]; then
    echo "No DISPLAY found, running in headless mode"
    HEADLESS_ARG="--headless"
else
    echo "DISPLAY detected: $DISPLAY"
fi

for subdir in "$INPUT_DIR"/*; do
    if [ ! -d "$subdir" ]; then
        echo "Skipping $subdir, not a directory"
        continue
    fi

    subdir_name=$(basename "$subdir")
    echo "Running FoundationPose on $subdir_name"
    # automatically get the mesh file, assuming it is the only file in "$INPUT_DIR/$subdir_name/mesh"
    # and has the extension ".obj"
    mesh_file=$(find "$INPUT_DIR/$subdir_name/mesh" -type f -name "*.obj" | head -n 1)

    python run_demo.py \
        --mesh_file "$mesh_file" \
        --test_scene_dir "$INPUT_DIR/$subdir_name" \
        --est_refine_iter 5 \
        --track_refine_iter 2 \
        --debug 2 \
        --prompts "${PROMPTS[@]}" \
        --map_to_table_frame \
        $HEADLESS_ARG

    yes | ffmpeg -framerate 20 -i "$INPUT_DIR/$subdir_name/track_vis/%04d.png" \
         -c:v mpeg4 -q:v 1 -pix_fmt yuv420p "$INPUT_DIR/$subdir_name/${subdir_name}_vis.mp4"

done
