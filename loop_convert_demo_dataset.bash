#!/usr/bin/env bash

# make sure the conda env is "demo_translate"
set -ex

# -------------------------------------------------------------------------------------------------------------
# Loop over block, cup and plate datasets with their respective prompts, meshes and output dirs
ROOT='/data3'
# ROOT=$HOME

DEMO_DATASET_DIRS=($ROOT/stack_three_blocks $ROOT/stack_three_cups $ROOT/stack_three_plates)
MESHS=(cube.obj cup.obj plate.obj)
OUTPUT_SUFFIX=(for_FoundationPose_blocks for_FoundationPose_cups for_FoundationPose_plates)
PROMPTS_LIST=(
  "red cube;blue cube;green cube"
  "orange cup;white cup that is NOT teal;teal cup"
  "orange plate;yellow plate;teal plate"
)

for i in "${!DEMO_DATASET_DIRS[@]}"; do
  DEMO_DATASET_DIR="${DEMO_DATASET_DIRS[$i]}"
  MESH_FILE="${MESHS[$i]}"
  OUTPUT_DIR="${DEMO_DATASET_DIR}/${OUTPUT_SUFFIX[$i]}"

  # split the semicolon-separated prompts into an array
  IFS=';' read -r -a PROMPTS <<< "${PROMPTS_LIST[$i]}"

  # ensure the root input dir exists
  if [ ! -d "$DEMO_DATASET_DIR" ]; then
      echo "Demo dataset directory does not exist: $DEMO_DATASET_DIR"
      continue
  fi

  # create the top-level output folder
  mkdir -p "$OUTPUT_DIR"

  # only loop through subdirs starting with "demonstration"
  for subdir in "$DEMO_DATASET_DIR"/demonstration*; do
    [ -d "$subdir" ] || continue

    subdir_name=$(basename "$subdir")
    this_out="${OUTPUT_DIR}/${subdir_name}"
    mkdir -p "$this_out"

    echo "Running FoundationPose on $subdir_name"

    python demo2dirs.py \
      --output_dir "$this_out" \
      --mesh_file "$DEMO_DATASET_DIR/$MESH_FILE" \
      --input_dir "$subdir"

    python $HOME/demo_translate/run_lang_sam.py \
      --input_dir "$this_out/rgb" \
      --prompts "${PROMPTS[@]}"
  done
done
