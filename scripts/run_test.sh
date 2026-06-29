#!/bin/bash

trap "exit" INT TERM

VIDEO_FILENAMES=(
   P30_110_9470_9497_right_mug_0645.mp4
)
INPUT_DIR=/home/b5db/jiahezhao25.b5db/jiahe/data/epic-grasps/epic-contact_2026-02-17_full
OUTPUT_DIR=/path/to/output

for VIDEO_FILE in "${VIDEO_FILENAMES[@]}"; do
    echo "Launching Srun for video: ${VIDEO_FILE}"
    srun --gpus=1 python -u demo_run.py \
        -i ${INPUT_DIR} \
        -o ${OUTPUT_DIR} \
        -s ${VIDEO_FILE}
done
