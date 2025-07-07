#!/usr/bin/env bash

# make sure the conda env is "demo_translate"
set -ex

# -------------------------------------------------------------------------------------------------------------



DEMO_DATASET_DIR="/data3/fp_data/stack_three_blocks"
PROMPTS=("red cube" "blue cube" "green cube" "Franka robot arm")  # for lang-sam segmentations. Assumes same CAD model for all
MESH_FILE="cube.obj"           # relative to DEMO_DATASET_DIR
OUTPUT_DIR="${DEMO_DATASET_DIR}/for_FoundationPose_blocks"
BOX_THRESHOLD=0.3
TEXT_THRESHOLD=0.25


# DEMO_DATASET_DIR="/data3/fp_data/stack_three_cups"
# PROMPTS=("orange cup" "white cup that is NOT teal" "teal cup" "Franka robot arm")  # for lang-sam segmentations. Assumes same CAD model for all
# MESH_FILE="cup.obj"           # relative to DEMO_DATASET_DIR
# OUTPUT_DIR="${DEMO_DATASET_DIR}/for_FoundationPose_cups"
# BOX_THRESHOLD=0.3
# TEXT_THRESHOLD=0.25


# DEMO_DATASET_DIR="/data3/fp_data/stack_three_plates"
# PROMPTS=("Franka robot arm" "yellow plate" "teal plate" "orange plate")  # for lang-sam segmentations. Assumes same CAD model for all
# MESH_FILE="plate.obj"           # relative to DEMO_DATASET_DIR
# OUTPUT_DIR="${DEMO_DATASET_DIR}/for_FoundationPose_plates"
# BOX_THRESHOLD=0.05
# TEXT_THRESHOLD=0.1



# -------------------------------------------------------------------------------------------------------------
# CAM_NUMBER=1 # foundation pose only needs to be run on one camera -- the one facing the april tag


# ensure the root input dir exists
if [ ! -d "$DEMO_DATASET_DIR" ]; then
    echo "Demo dataset directory does not exist: $DEMO_DATASET_DIR"
    exit 1
fi

# create the top‐level output folder
mkdir -p "$OUTPUT_DIR"

# only loop through subdirs starting with "demonstration"
for subdir in "$DEMO_DATASET_DIR"/demonstration*; do
  # ensure it actually exists and is a directory
  [ -d "$subdir" ] || continue

  subdir_name=$(basename "$subdir")

  # make a per‐camera output folder
  this_out="${OUTPUT_DIR}/${subdir_name}"
  mkdir -p "$this_out"

  # echo "Running FoundationPose on $subdir_name for camera $cam_num"
  # python demo2dirs.py \
  #   --output_dir "$this_out" \
  #   --mesh_file "$DEMO_DATASET_DIR/$MESH_FILE" \
  #   --input_dir "$subdir"
  # # run lang-sam to get masks for first images
  # python $HOME/demo_translate/run_lang_sam.py \
  # --input_dir "$this_out/rgb" \
  # --prompts "${PROMPTS[@]}" \
  # --annotation_dir "$subdir" \
  # --box_threshold "$BOX_THRESHOLD" \
  # --text_threshold "$TEXT_THRESHOLD" \

  # === ADDITION: Generate 3x4 tiled video ===
  echo "Creating 3x4 mask overlay video from outputs..."

  pushd "$this_out" > /dev/null

  TILE_DIRS=($(find . -maxdepth 1 -type d -name 'rgb_cam_*_masks*' | sort))
  # echo "Tiling directories: ${TILE_DIRS[@]}"
  mkdir -p tiled_frames
  python $HOME/FoundationPose/make_seg_video.py \
    --input_dirs "${TILE_DIRS[@]}" \
    --output_dir tiled_frames \
    --tile_shape 3 4 \
    --frame_rate 10
  ffmpeg -y -r 10 -i tiled_frames/%04d.png -c:v mpeg4 -q:v 5 -pix_fmt yuv420p masks_${subdir_name}.mp4

  popd > /dev/null
  exit
done


