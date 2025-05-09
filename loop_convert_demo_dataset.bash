#!/usr/bin/env bash

# make sure the conda env is "demo_translate"
set -ex

# -------------------------------------------------------------------------------------------------------------
# Loop over block, cup and plate datasets with their respective prompts, meshes and output dirs
ROOT='/data3/fp_data'
# ROOT=$HOME

DEMO_DATASET_DIRS=($ROOT/stack_three_blocks $ROOT/stack_three_cups $ROOT/stack_three_plates)
MESHS=(cube.obj cup.obj plate.obj)
OUTPUT_SUFFIX=(for_FoundationPose_blocks for_FoundationPose_cups for_FoundationPose_plates)
PROMPTS_LIST=(
  "Franka robot arm;red cube;blue cube;green cube"
  "Franka robot arm;orange cup;white cup that is NOT teal;teal cup"
  "Franka robot arm;orange plate;yellow plate;teal plate"
)
BOX_THRESHOLDS=(0.3 0.3 0.05)
TEXT_THRESHOLDS=(0.25 0.25 0.1)

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
      --prompts "${PROMPTS[@]}" \
      --box_threshold "${BOX_THRESHOLDS[$i]}" \
      --text_threshold "${TEXT_THRESHOLDS[$i]}" \
      --annotation_dir "$subdir"
    # === ADDITION: Generate 3x3 tiled video ===
    echo "Creating 3x3 mask overlay video from outputs..."

    pushd "$this_out" > /dev/null

    TILE_DIRS=($(find . -maxdepth 1 -type d -name 'rgb_cam_*_masks*' | sort))

    mkdir -p tiled_frames

    for i in $(find "${TILE_DIRS[0]}" -maxdepth 1 -type f -name '*_overlay.png' | sed 's/.*\///;s/_overlay\.png//' | sort); do
      convert \( ${TILE_DIRS[0]}/${i}_overlay.png \
            ${TILE_DIRS[1]}/${i}_overlay.png \
            ${TILE_DIRS[2]}/${i}_overlay.png \
            ${TILE_DIRS[3]}/${i}_overlay.png +append \) \
          \( ${TILE_DIRS[4]}/${i}_overlay.png \
            ${TILE_DIRS[5]}/${i}_overlay.png \
            ${TILE_DIRS[6]}/${i}_overlay.png \
            ${TILE_DIRS[7]}/${i}_overlay.png +append \) \
          \( ${TILE_DIRS[8]}/${i}_overlay.png \
            ${TILE_DIRS[9]}/${i}_overlay.png \
            ${TILE_DIRS[10]}/${i}_overlay.png \
            ${TILE_DIRS[11]}/${i}_overlay.png +append \) \
              -append tiled_frames/${i}.png || break

      convert tiled_frames/${i}.png -resize 50% tiled_frames/${i}.png
    done

    ffmpeg -y -r 10 -i tiled_frames/%04d.png -c:v mpeg4 -q:v 5 -pix_fmt yuv420p masks_${subdir_name}.mp4

    popd > /dev/null
  done
done
