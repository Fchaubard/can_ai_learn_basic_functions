#!/bin/bash

##############################################################################
# ablation_study_run.sh
#
# Ablation study script for MeZO Adaptive Sampler.
# This script sweeps over a number of hyperparameters including:
#   - TASK, ARCH, MODEL_SCALE, OPTIMIZER, INPUT_SAMPLE_LENGTH,
#     MICRO_BATCH_SIZE, MACRO_BATCH_SIZE, LEARNING_RATE, EPSILON,
#     EPS_LR_RATIO, MAX_NUM, WARMUP_STEP, WEIGHT_DECAY, GRAD_CLIP, PAD_BIAS,
#   - n_perturbations (5, 10, 20, 40)
#   - use_same_eps_for_all_perturbations (True, False)
#       - When False, additionally: eps_schedule_multiplier (2, 3)
#         and use_same_probe_for_all_perturbations (True, False)
#   - aggregation_method ("average", "max", "weighted_average")
#
# Each configuration is launched in a detached screen session.
##############################################################################

# 1) Kill existing screen sessions with prefix "MeZO_"
echo "[INFO] Killing existing screen sessions named 'R*'..."
screen -ls | grep '\.R' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting ablation study runs..."

# Define hyperparameter arrays (adjust these values as needed):
TASKS=("owt")
ARCHITECTURES=("dnc")
OPTIMIZERS=("mezo_adaptive_sampling_layerwise_one_way") # "mezo_adaptive_sampling_fast")
# MODEL_SCALES=(1)
MODEL_SCALES=(100 200)

hidden_size=10
memory_size=10
head_size=10
num_heads=1
input_dim=32
variance_reduction=1.
adam_variance="False" 

INPUT_SAMPLE_LENGTHS=(150)
MICRO_BATCH_SIZES=(32)
MACRO_BATCH_SIZES=(1)
LEARNING_RATES=(0.001)
EPSILONS=(0.01)
EPS_LR_RATIOS=(0.5)
MAX_NUMS=(120)
WARMUP_STEPS=(100)
WEIGHT_DECAYS=(0.000001)
GRAD_CLIPS=(0)
PAD_BIASES=(0.0)



# New ablation parameters:
# NUM_PERTURBATIONS=(5 10 20 40)
NUM_PERTURBATIONS=(3 5 10 20)
USE_SAME_EPS_VALUES=("False")
EPS_MULTIPLIERS=(1)
USE_SAME_PROBE_VALUES=("True")
AGGREGATION_METHODS=("average" "max" "weighted_average")
# AGGREGATION_METHODS=("average")

# Other configurations:
LOG_INTERVAL=100
MAX_ITERS=50000
WANDB_PROJ="DNC-MODELSIZE-ABLATION"
RUN_PREFIX="RX"

# Function to truncate a long session name (if needed)
truncate_name() {
  echo "$1" | cut -c1-65
}

# --- Loop over the original hyperparameters ---
for TASK in "${TASKS[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    for MODEL_SCALE in "${MODEL_SCALES[@]}"; do
      for OPTIMIZER_TYPE in "${OPTIMIZERS[@]}"; do
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
                              
                              # --- Compute model scale–dependent parameters ---
                              this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                              this_memory_size=$(( memory_size * MODEL_SCALE ))
                              this_head_size=$(( head_size * MODEL_SCALE ))
                              this_num_head=$(( num_heads * MODEL_SCALE ))
                              
                              # --- Loop over new ablation parameters ---
                              for numPert in "${NUM_PERTURBATIONS[@]}"; do
                                for useSameEps in "${USE_SAME_EPS_VALUES[@]}"; do
                                  if [ "$useSameEps" = "True" ]; then
                                    # When use_same_eps_for_all_perturbations is True,
                                    # force use_same_probe_for_all_perturbations to be False.
                                    useSameProbe="False"
                                    # Determine extra flags based on optimizer type.
                                    if [[ "$OPTIMIZER_TYPE" == mezo* ]]; then
                                      BLAH="mezo"
                                      FIXED_FLAG="--fixed_size_perturbation"
                                      EXTRA_FLAGS="--cosine_lr --optimizer ${BLAH}"
                                    else
                                      BLAH="sgd"
                                      FIXED_FLAG=""
                                      EXTRA_FLAGS=""
                                    fi
                                    
                                    for agg in "${AGGREGATION_METHODS[@]}"; do
                                      # Construct run name including model scale parameters.
                                      RUN_NAME_BASE="${RUN_PREFIX}_${ARCH}_${OPTIMIZER_TYPE}_scale${MODEL_SCALE}_iters${MAX_ITERS}_lr${LR}_eps${EPS}_elratio${RATIO}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}_npert${numPert}_usesameeps${useSameEps}_usesameprobe${useSameProbe}_agg${agg}"
                                      RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                      echo "[INFO] Launching screen session: $RUN_NAME"
                                      
                                      screen -dmS "$RUN_NAME" bash -c "
                                        echo '[INFO] Starting run: $RUN_NAME';
                                        export WANDB_RUN_NAME=$RUN_NAME;
                                        python ntm_with_modern_training_runs.py \
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
                                          --max_num ${MAX_NUM} \
                                          --warmup_steps ${WARMUP_STEP} \
                                          --weight_decay ${WEIGHT_DECAY} \
                                          --grad_clip ${GRAD_CLIP} \
                                          --pad_bias ${PAD_BIAS} \
                                          --mezo_flavor ${OPTIMIZER_TYPE} \
                                          ${EXTRA_FLAGS} ${FIXED_FLAG} \
                                          --num_perturbations ${numPert} \
                                          --aggregation_method ${agg} \
                                          --use_same_eps_for_all_perturbations \
                                          --variance_reduction $variance_reduction \
                                          --wandb_proj ${WANDB_PROJ} \
                                          --wandb_run_name ${RUN_NAME};
                                          
                                        echo '[INFO] Finished run: $RUN_NAME';
                                        exec bash
                                      "
                                      sleep 4
                                    done
                                  else
                                    # use_same_eps_for_all_perturbations is False.
                                    for epsMult in "${EPS_MULTIPLIERS[@]}"; do
                                      for useSameProbe in "${USE_SAME_PROBE_VALUES[@]}"; do
                                        if [[ "$OPTIMIZER_TYPE" == mezo* ]]; then
                                          BLAH="mezo"
                                          # FIXED_FLAG="--fixed_size_perturbation"
                                          EXTRA_FLAGS="--cosine_lr --optimizer ${BLAH}"
                                        else
                                          BLAH="sgd"
                                          FIXED_FLAG=""
                                          EXTRA_FLAGS=""
                                        fi
                                        
                                        for agg in "${AGGREGATION_METHODS[@]}"; do
                                          RUN_NAME_BASE="${RUN_PREFIX}_${ARCH}_${OPTIMIZER_TYPE}_scale${MODEL_SCALE}_iters${MAX_ITERS}_lr${LR}_eps${EPS}_elratio${RATIO}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}_npert${numPert}_usesameeps${useSameEps}_epsmult${epsMult}_usesameprobe${useSameProbe}_agg${agg}"
                                          RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                          echo "[INFO] Launching screen session: $RUN_NAME"
                                          CMD="python ntm_with_modern_training_runs.py \
                                            --task ${TASK} \
                                            --arch ${ARCH} \
                                            --input_sample_length ${INPUT_SAMPLE_LENGTH} \
                                            --hidden_size ${this_hidden_size} \
                                            --memory_size ${this_memory_size} \
                                            --head_size ${this_head_size} \
                                            --num_heads ${num_heads} \
                                            --input_size ${input_dim} \
                                            --micro_batch_size ${MICRO_BS} \
                                            --macro_batch_size ${MACRO_BS} \
                                            --max_iters ${MAX_ITERS} \
                                            --log_interval ${LOG_INTERVAL} \
                                            --learning_rate ${LR} \
                                            --tie_epsilon_to_lr_ratio ${RATIO} \
                                            --max_num ${MAX_NUM} \
                                            --warmup_steps ${WARMUP_STEP} \
                                            --weight_decay ${WEIGHT_DECAY} \
                                            --grad_clip ${GRAD_CLIP} \
                                            --pad_bias ${PAD_BIAS} \
                                            --mezo_flavor ${OPTIMIZER_TYPE} \
                                            ${EXTRA_FLAGS} ${FIXED_FLAG} \
                                            --num_perturbations ${numPert} \
                                            --variance_reduction $variance_reduction \
                                            --aggregation_method ${agg} \
                                            --eps_schedule_multiplier ${epsMult}"
                                          if [ "$useSameProbe" = "True" ]; then
                                            CMD="$CMD --use_same_probe_for_all_perturbations"
                                          fi
                                          if [ "$adam_variance" = "True" ]; then
                                            CMD="$CMD --adam_variance"
                                          fi
                                          CMD="$CMD --wandb_proj ${WANDB_PROJ} --wandb_run_name ${RUN_NAME}"
                                          screen -dmS "$RUN_NAME" bash -c "
                                            echo '[INFO] Starting run: $RUN_NAME';
                                            $CMD;
                                            echo '[INFO] Finished run: $RUN_NAME';
                                            exec bash
                                          "
                                          sleep 4
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
