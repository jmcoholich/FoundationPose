#!/bin/bash
set -e

TASK_NAMES=("stack_three_blocks" "stack_three_cups" "stack_three_plates")
H5_ROOT='/data3/fp_data'
FP_DIRS=("for_FoundationPose_blocks" "for_FoundationPose_cups" "for_FoundationPose_plates")
OBJS=("blocks" "cups" "plates")

# output dirs are OUTPUT_task_name
OUTPUT_DIRS=()
for task in "${TASK_NAMES[@]}"; do
    OUTPUT_DIRS+=("$H5_ROOT/OUTPUT_$task")
done

# # raise an error if the directories already exist; otherwise create them
# echo "Checking if output directories already exist..."
# for dir in "${OUTPUT_DIRS[@]}"; do
#     if [ -d "$dir" ]; then
#         echo "Error: Directory $dir already exists. Please remove it or choose a different name."
#         exit 1
#     else
#         echo "Creating directory: $dir"
#         mkdir -p "$dir"
#     fi
# done

# # copy all h5 data over to output dirs
# echo "Copying H5 files to output directories..."
# for i in "${!OUTPUT_DIRS[@]}"; do
#     OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
#     TASK_NAME="${TASK_NAMES[$i]}"

#     echo "Processing task: $TASK_NAME"
#     echo "Finding H5 files in $H5_ROOT/$TASK_NAME..."
#     # echo $(find "$H5_ROOT/$TASK_NAME" -type f -name "*demo*.h5")
#     find "$H5_ROOT/$TASK_NAME" -type f -name "*demo*.h5" | xargs -P 4 -I {} rsync -a {} "$OUTPUT_DIR"
#     # exit 0
# done

# # copy all mp4 files over to output_dirs
# echo "Copying MP4 files to output directories..."
# for i in "${!OUTPUT_DIRS[@]}"; do
#     OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
#     FP_DIR="${FP_DIRS[$i]}"

#     echo "Processing FoundationPose directory: $FP_DIR"
#     echo "Finding MP4 files in $FP_DIR..."
#     find -L "$FP_DIR" -type f -name "*.mp4" | xargs -P 4 -I {} rsync -a {} "$OUTPUT_DIR"
# done

set -x
# run the update_h5.py script for each output directory
echo "Running update_h5.py script for each output directory..."
for i in "${!OUTPUT_DIRS[@]}"; do
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    FP_DIR="${FP_DIRS[$i]}"
    OBJ_TYPE="${OBJS[$i]}"

    echo "Updating H5 files in $OUTPUT_DIR for object type: $OBJ_TYPE"
    python update_h5.py --h5_dir "$OUTPUT_DIR" --obj "$OBJ_TYPE" --foundationPose_dir "$FP_DIR"
done
