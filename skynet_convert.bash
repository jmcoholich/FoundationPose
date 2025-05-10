#!/usr/bin/env bash
#SBATCH --job-name=foundationpose_array
#SBATCH --partition=kira-lab
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-60
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -ex

# activate your conda env
eval "$(conda shell.bash hook)"
conda activate demo_translate

# -------------------------------------------------------------------------------------------------------------
ROOT=$HOME/fp_data

# parallel arrays for dataset configs
DEMO_DATASET_DIRS=( \
  $ROOT/stack_three_blocks \
  $ROOT/stack_three_cups \
  $ROOT/stack_three_plates \
)
MESHS=(cube.obj cup.obj plate.obj)
OUTPUT_SUFFIX=(for_FoundationPose_blocks for_FoundationPose_cups for_FoundationPose_plates)
PROMPTS_LIST=( \
  "Franka robot arm;red cube;blue cube;green cube" \
  "Franka robot arm;orange cup;white cup that is NOT teal;teal cup" \
  "Franka robot arm;orange plate;yellow plate;teal plate" \
)
BOX_THRESHOLDS=(0.3 0.3 0.05)
TEXT_THRESHOLDS=(0.25 0.25 0.1)

counter=0

# loop over dataset types
for i in "${!DEMO_DATASET_DIRS[@]}"; do
  DEMO_DATASET_DIR="${DEMO_DATASET_DIRS[$i]}"
  MESH_FILE="${MESHS[$i]}"
  OUTPUT_DIR="${DEMO_DATASET_DIR}/${OUTPUT_SUFFIX[$i]}"

  # split the semicolon-separated prompts into an array
  IFS=';' read -r -a PROMPTS <<< "${PROMPTS_LIST[$i]}"

  # skip if missing
  [ -d "$DEMO_DATASET_DIR" ] || continue
  mkdir -p "$OUTPUT_DIR"

  # only look at demonstration* subdirs
  for subdir in "$DEMO_DATASET_DIR"/demonstration*; do
    [ -d "$subdir" ] || continue

    counter=$((counter + 1))
    # only run the one matching our array index
    if [ "$counter" -ne "$SLURM_ARRAY_TASK_ID" ]; then
      continue
    fi

    subdir_name=$(basename "$subdir")
    this_out="${OUTPUT_DIR}/${subdir_name}"
    mkdir -p "$this_out"

    echo "[$SLURM_ARRAY_TASK_ID] Processing $DEMO_DATASET_DIR → $subdir_name"

    python demo2dirs.py \
      --output_dir "$this_out" \
      --mesh_file "$DEMO_DATASET_DIR/$MESH_FILE" \
      --input_dir "$subdir"

    python $HOME/demo_translate/run_lang_sam.py \
      --input_dir "$this_out/rgb" \
      --prompts "${PROMPTS[@]}" \
      --bs 3 \
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
    # get demo_dataset_dir basename
    DIR_BASENAME=$(basename "$DEMO_DATASET_DIR")
    # sync results back to my PC
    for attempt in {1..3}; do
      # make sure the remote directory exists
      ssh jcoholich@143.215.128.197 "mkdir -p /data3/fp_data/$DIR_BASENAME/${OUTPUT_SUFFIX[$i]}/"
      # sync the output directory
      rsync -ahP $this_out jcoholich@143.215.128.197:/data3/fp_data/$DIR_BASENAME/${OUTPUT_SUFFIX[$i]}/
      sleep 5
    done
    # once we’ve matched, break both loops
    break 2
  done
done
