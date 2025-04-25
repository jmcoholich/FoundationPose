#!/bin/bash
set -ex

yes | conda install ffmpeg
INPUT_DIR="for_FoundationPose"
PROMPTS=("blue_cube" "red_cube" "green_cube")  # just to get mask dirs

# run script for every subdir in the input directory
#   parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
#   parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
#   parser.add_argument('--est_refine_iter', type=int, default=5)
#   parser.add_argument('--track_refine_iter', type=int, default=2)
#   parser.add_argument('--debug', type=int, default=1)
#   parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')

for subdir in "$INPUT_DIR"/*; do
    # ensure it actually exists and is a directory. If it doesn't print a message
    if [ ! -d "$subdir" ]; then
        echo "Skipping $subdir, not a directory"
        continue
    fi

    subdir_name=$(basename "$subdir")
    echo "Running FoundationPose on $subdir_name"
    # automatically get the mesh file, assuming it is the only file in "$INPUT_DIR/$subdir_name/mesh"
    # and has the extension ".obj"
    mesh_file=$(find "$INPUT_DIR/$subdir_name/mesh" -type f -name "*.obj" | head -n 1)
    # dirs are messed up, debug
    # run FoundationPose
    # for PROMPT in "${PROMPTS[@]}"; do
    python run_demo.py \
        --mesh_file "$mesh_file" \
        --test_scene_dir "$INPUT_DIR/$subdir_name" \
        --est_refine_iter 5 \
        --track_refine_iter 2 \
        --debug 2 \
        --prompts "${PROMPTS[@]}" \
        --map_to_table_frame
        # --debug_dir "$INPUT_DIR/$subdir_name/outputs_$PROMPT" \
    # done
    yes | ffmpeg -framerate 15 -i $INPUT_DIR/$subdir_name/track_vis/%04d.png -c:v mpeg4 -q:v 1 -pix_fmt yuv420p $INPUT_DIR/$subdir_name/vis.mp4
done