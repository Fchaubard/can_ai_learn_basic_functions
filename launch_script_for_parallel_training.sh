#!/bin/bash

##############################################################################
# run_distributed_training.sh
#
# Distributed training script for MeZO with torch.distributed.
# This script sweeps over various hyperparameters similar to the original
# ablation_study_run.sh but adapted for distributed training.
##############################################################################

# Kill existing screen sessions with prefix
echo "[INFO] Killing existing screen sessions named 'DIST*'..."
screen -ls | grep '\.DIST' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting distributed training runs..."

# Define hyperparameter arrays (adjust these values as needed):
TASKS=("owt")
ARCHITECTURES=("dnc")

# Always using mezo_adaptive_sampling_fast_one_way as requested
OPTIMIZER="mezo_adaptive_sampling_fast_one_way"

MODEL_SCALES=(3000) #2500 max? but hanging

# Base model architecture parameters
hidden_size=10
memory_size=10
head_size=10 
num_heads=1
input_dim=32
variance_reduction=1.

INPUT_SAMPLE_LENGTHS=(150)
MICRO_BATCH_SIZES=(32)
MACRO_BATCH_SIZES=(1)

LEARNING_RATES=(0.001) 
EPSILONS=(0.01)
EPS_LR_RATIOS=(1.0)
MAX_NUMS=(120)
WARMUP_STEPS=(100)
WEIGHT_DECAYS=(0.)
GRAD_CLIPS=(0)
PAD_BIASES=(0.0)

# Probe dropout rates to try
DROPOUTRATES=(.99)

# MeZO specific parameters
NUM_PERTURBATIONS=(7)
USE_SAME_EPS_VALUES=("False")
EPS_MULTIPLIERS=(1)
USE_SAME_PROBE_VALUES=("False")
AGGREGATION_METHODS=("average")

# Other configurations:
LOG_INTERVAL=100
MAX_ITERS=1000
WANDB_PROJ="DNC-SINGLE-BATCH-MEMORIZE"
RUN_PREFIX="DIST_1way"

# Function to truncate a long session name (if needed)
truncate_name() {
  echo "$1" | cut -c1-65
}

# Function to get available GPU count
get_gpu_count() {
  nvidia-smi --list-gpus | wc -l
}

# Get total available GPUs
TOTAL_GPUS=$(get_gpu_count)
echo "[INFO] Total available GPUs: $TOTAL_GPUS"

# Loop over the original hyperparameters
for TASK in "${TASKS[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
      for INPUT_SAMPLE_LENGTH in "${INPUT_SAMPLE_LENGTHS[@]}"; do
        for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
          for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
              for EPS in "${EPSILONS[@]}"; do
                for RATIO in "${EPS_LR_RATIOS[@]}"; do
                  for MAX_NUM in "${MAX_NUMS[@]}"; do
                    for WARMUP_STEP in "${WARMUP_STEPS[@]}"; do
                      for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                        for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
                          for PAD_BIAS in "${PAD_BIASES[@]}"; do
                            for DROPOUTRATE in "${DROPOUTRATES[@]}"; do
                              
                              # Compute model scale–dependent parameters
                              this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                              this_memory_size=$(( memory_size * MODEL_SCALE ))
                              this_head_size=$(( head_size  ))
                              this_num_head=$num_heads
                              
                              # Loop over new ablation parameters
                              for numPert in "${NUM_PERTURBATIONS[@]}"; do
                                # Ensure we use at most the available GPUs
                                if [ $((numPert + 1)) -le $TOTAL_GPUS ]; then
                                  USE_GPUS=$((numPert + 1))
                                else
                                  echo "[WARNING] Not enough GPUs for $numPert perturbations. Using $TOTAL_GPUS GPUs."
                                  USE_GPUS=$TOTAL_GPUS
                                fi
                                
                                for useSameEps in "${USE_SAME_EPS_VALUES[@]}"; do
                                  if [ "$useSameEps" = "True" ]; then
                                    # When use_same_eps_for_all_perturbations is True
                                    useSameProbe="False"
                                    EXTRA_FLAGS="--cosine_lr --use_same_eps_for_all_perturbations"
                                    
                                    for agg in "${AGGREGATION_METHODS[@]}"; do
                                      # Construct run name 
                                      RUN_NAME_BASE="${RUN_PREFIX}_${ARCH}_scale${MODEL_SCALE}_DOR${DROPOUTRATE}_lr${LR}_eps${EPS}_elratio${RATIO}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}_npert${numPert}_gpus${USE_GPUS}_usesameeps${useSameEps}_usesameprobe${useSameProbe}_agg${agg}"
                                      RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                      echo "[INFO] Launching screen session: $RUN_NAME with $USE_GPUS GPUs"
                                      
                                      screen -dmS "$RUN_NAME" bash -c "
                                        echo '[INFO] Starting run: $RUN_NAME';
                                        export WANDB_RUN_NAME=$RUN_NAME;
                                        python -m torch.distributed.launch --nproc_per_node=${USE_GPUS} dnc_parallel_training.py \
                                          --task ${TASK} \
                                          --arch ${ARCH} \
                                          --input_sample_length ${INPUT_SAMPLE_LENGTH} \
                                          --hidden_size ${this_hidden_size} \
                                          --memory_size ${this_memory_size} \
                                          --head_size ${this_head_size} \
                                          --num_heads ${this_num_head} \
                                          --input_size ${input_dim} \
                                          --micro_batch_size ${MICRO_BS} \
                                          --macro_batch_size ${MACRO_BS} \
                                          --max_iters ${MAX_ITERS} \
                                          --log_interval ${LOG_INTERVAL} \
                                          --learning_rate ${LR} \
                                          --tie_epsilon_to_lr_ratio ${RATIO} \
                                          --epsilon ${EPS} \
                                          --max_num ${MAX_NUM} \
                                          --warmup_steps ${WARMUP_STEP} \
                                          --weight_decay ${WEIGHT_DECAY} \
                                          --grad_clip ${GRAD_CLIP} \
                                          --pad_bias ${PAD_BIAS} \
                                          --mezo_flavor ${OPTIMIZER} \
                                          ${EXTRA_FLAGS} \
                                          --num_perturbations ${numPert} \
                                          --aggregation_method ${agg} \
                                          --probe_dropout_rate ${DROPOUTRATE} \
                                          --variance_reduction $variance_reduction \
                                          --wandb_proj ${WANDB_PROJ} \
                                          --reset_solver_after_plateau 0 \
                                          --cosine_windowing \
                                          --overfit_to_one_batch_flag \
                                          --wandb_run_name ${RUN_NAME};
                                        echo '[INFO] Finished run: $RUN_NAME';
                                        exec bash
                                      "
                                      sleep 10
                                    done
                                  else
                                    # use_same_eps_for_all_perturbations is False
                                    for epsMult in "${EPS_MULTIPLIERS[@]}"; do
                                      for useSameProbe in "${USE_SAME_PROBE_VALUES[@]}"; do
                                        EXTRA_FLAGS="--cosine_lr --eps_schedule_multiplier ${epsMult}"
                                        if [ "$useSameProbe" = "True" ]; then
                                          EXTRA_FLAGS="$EXTRA_FLAGS --use_same_probe_for_all_perturbations"
                                        fi
                                        
                                        for agg in "${AGGREGATION_METHODS[@]}"; do
                                          RUN_NAME_BASE="${RUN_PREFIX}_${ARCH}_scale${MODEL_SCALE}_DOR${DROPOUTRATE}_lr${LR}_eps${EPS}_elratio${RATIO}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}_npert${numPert}_gpus${USE_GPUS}_usesameeps${useSameEps}_usesameprobe${useSameProbe}_agg${agg}"
                                          RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                          echo "[INFO] Launching screen session: $RUN_NAME with $USE_GPUS GPUs"
                                          
                                          screen -dmS "$RUN_NAME" bash -c "
                                            echo '[INFO] Starting run: $RUN_NAME';
                                            export WANDB_RUN_NAME=$RUN_NAME;
                                            python -m torch.distributed.launch --nproc_per_node=${USE_GPUS} dnc_parallel_training.py \
                                              --task ${TASK} \
                                              --arch ${ARCH} \
                                              --input_sample_length ${INPUT_SAMPLE_LENGTH} \
                                              --hidden_size ${this_hidden_size} \
                                              --memory_size ${this_memory_size} \
                                              --head_size ${this_head_size} \
                                              --num_heads ${this_num_head} \
                                              --input_size ${input_dim} \
                                              --micro_batch_size ${MICRO_BS} \
                                              --macro_batch_size ${MACRO_BS} \
                                              --max_iters ${MAX_ITERS} \
                                              --log_interval ${LOG_INTERVAL} \
                                              --learning_rate ${LR} \
                                              --tie_epsilon_to_lr_ratio ${RATIO} \
                                              --epsilon ${EPS} \
                                              --max_num ${MAX_NUM} \
                                              --warmup_steps ${WARMUP_STEP} \
                                              --weight_decay ${WEIGHT_DECAY} \
                                              --grad_clip ${GRAD_CLIP} \
                                              --pad_bias ${PAD_BIAS} \
                                              --mezo_flavor ${OPTIMIZER} \
                                              ${EXTRA_FLAGS} \
                                              --num_perturbations ${numPert} \
                                              --aggregation_method ${agg} \
                                              --probe_dropout_rate ${DROPOUTRATE} \
                                              --variance_reduction $variance_reduction \
                                              --wandb_proj ${WANDB_PROJ} \
                                              --reset_solver_after_plateau 0 \
                                              --cosine_windowing \
                                              --overfit_to_one_batch_flag \
                                              --wandb_run_name ${RUN_NAME};
                                            echo '[INFO] Finished run: $RUN_NAME';
                                            exec bash
                                          "
                                          sleep 10
                                        done
                                      done
                                    done
                                  fi
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "[INFO] Done launching all screen sessions."
echo "[INFO] Use 'screen -ls' to see running sessions."
echo "[INFO] Use 'screen -r SESSION_NAME' to attach to a session."
