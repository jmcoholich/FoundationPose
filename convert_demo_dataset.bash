#!/usr/bin/env bash

# make sure the conda env is "demo_translate"
set -ex
DEMO_DATASET_DIR="/data3/TEST_demo_dataset"
PROMPTS=("blue cube" "red cube" "green cube")  # for lang-sam segmentations. Assumes same CAD model for all
MESH_FILE="cube.obj"           # relative to DEMO_DATASET_DIR
OUTPUT_DIR="${DEMO_DATASET_DIR}/for_FoundationPose"
CAM_NUMBER=1 # foundation pose only needs to be run on one camera -- the one facing the april tag

# get the home directory of the current user
USER_HOME=$(eval echo "~$USER")

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

  echo "Running FoundationPose on $subdir_name for camera $cam_num"
  python demo2dirs.py \
    --cam_num "$CAM_NUMBER" \
    --output_dir "$this_out" \
    --mesh_file "$DEMO_DATASET_DIR/$MESH_FILE" \
    --input_dir "$subdir"
  for PROMPT in "${PROMPTS[@]}"; do
    echo "Running lang-sam on $subdir_name for camera $CAM_NUMBER with prompt: $PROMPT"
      # run lang-sam to get masks for first images
      python $USER_HOME/demo_translate/run_lang_sam.py \
      --input_dir "$this_out/rgb" \
      --output_dir "$this_out/masks_$PROMPT"  \
      --prompt "$PROMPT"
  done
done


