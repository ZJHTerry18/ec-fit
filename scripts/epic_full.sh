#!/bin/bash
#SBATCH --job-name=parallel_ecfit_epic
#SBATCH --nodes=20
#SBATCH --ntasks=1280
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/epic-%j.out
#SBATCH --error=logs/epic-%j.err
#SBATCH --mail-user=jiahe.zhao@bristol.ac.uk
#SBATCH --mail-type=ALL

GPUS_PER_NODE=4

source ~/.bashrc
conda activate ecfit
mkdir -p logs

TOTAL_SAMPLES=2400
INPUT_DIR="/path/to/epic_contact_videos"
OUTPUT_DIR="/.../ecfit_output_[template]"

N_TASKS=$SLURM_NTASKS
if [ -z "$N_TASKS" ]; then
    echo "Error: SLURM_NTASKS not set. Did you run this script outside of Slurm?"
    exit 1
fi
CHUNK_SIZE=$((( TOTAL_SAMPLES / N_TASKS ) + 1))

echo "Total Samples: $TOTAL_SAMPLES"
echo "Total Tasks: $N_TASKS"
echo "Chunk Size (samples per task): $CHUNK_SIZE"
echo "Starting job on $SLURM_NNODES nodes."
echo "Start Time: $(date +%Y-%m-%d\ %H:%M:%S)"

srun bash -c "
    GLOBAL_RANK=\$SLURM_PROCID
    LOCAL_RANK=\$SLURM_LOCALID
    DEVICE_ID=\$(( LOCAL_RANK % $GPUS_PER_NODE ))
    export CUDA_VISIBLE_DEVICES=\$DEVICE_ID
    START_IDX=\$(( GLOBAL_RANK * $CHUNK_SIZE ))
    END_IDX=\$(( START_IDX + $CHUNK_SIZE ))

    echo \"Task \$GLOBAL_RANK (Node: \$LOCAL_RANK) using GPU: \$DEVICE_ID assigned range \$START_IDX to \$END_IDX\"

    python batch_run_generic.py \
        -i $INPUT_DIR -o $OUTPUT_DIR \
        --start \$START_IDX --end \$END_IDX
"

echo "End Time: $(date +%Y-%m-%d\ %H:%M:%S)"
