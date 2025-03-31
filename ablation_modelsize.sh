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
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting ablation study runs..."


########## RUN PREFIX #############
# RUN_PREFIX="QBaseline_SGD" 
# RUN_PREFIX="QBaseline_SPSA" 
# RUN_PREFIX="QQBaseline_MEZO" 
RUN_PREFIX="QZP_1way_noadapt" 
# RUN_PREFIX="QZP_2way" 
# CD, vanilla SGD, no DOR, one way and two way on this, all overfit


# Define hyperparameter arrays (adjust these values as needed):
TASKS=("owt")
ARCHITECTURES=("dnc")

# OPTIMIZERS=("mezo_single" "sgd" "mezo_layerwise")

OPTIMIZERS=("mezo_single")

# MODEL_SCALES=(1 2 4 8 16 32 64 128 256) 
# MODEL_SCALES=(64 ) 
MODEL_SCALES=(64 32)

hidden_size=128
memory_size=128
head_size=128
num_heads=1
input_dim=128
variance_reduction=1.

INPUT_SAMPLE_LENGTHS=(10 100)
MICRO_BATCH_SIZES=(1)
MACRO_BATCH_SIZES=(1)

LEARNING_RATES=(0.0001 0.001) 
EPSILONS=(0.001)
EPS_LR_RATIOS=(1.0)
MAX_NUMS=(120)
WARMUP_STEPS=(1)
WEIGHT_DECAYS=(0.)
GRAD_CLIPS=(0)
PAD_BIASES=(0.0)

# DROPOUTRATES=(0 .1 .2 .4 .6 .8 .9 .95 .99 .999)
# DROPOUTRATES=(.999 .99 .95 .9 .8 .6 .4 .2 .1 0.)
DROPOUTRATES=(0.999)

# DROPOUTRATES=(.6 .8 .9 .99 .999)
# DROPOUTRATES=(0.999)

# New ablation parameters:
# NUM_PERTURBATIONS=(5 10 20 40)
NUM_PERTURBATIONS=(27) #27
EPS_MULTIPLIERS=(1)
AGGREGATION_METHODS=("average")
# ONE_WAYS=(true false)
# ADAPTIVES=(true false)
# OVERFITS=(true false)

ONE_WAYS=(true )
ADAPTIVES=(false)
OVERFITS=(true)

# AGGREGATION_METHODS=("average")

# Other configurations:
LOG_INTERVAL=10
MAX_ITERS=500


# SOLVERS=("adam" "vanilla_sgd")
SOLVERS=("adam")
ADAM_BETA1=0.9
ADAM_BETA2=0.9
# ADAM_BETA1=0.0
# ADAM_BETA2=0.0

WANDB_PROJ="ZeroProp_Final_Runs2"

# RUN_PREFIX="Baseline_ofit_"
# Function to truncate a long session name (if needed)
truncate_name() {
  echo "$1" | cut -c1-65
}

# --- Detect Available GPUs ---
echo "[INFO] Detecting available GPUs..."
GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected by nvidia-smi. Exiting."
    exit 1
fi
echo "[INFO] Detected ${NUM_GPUS} GPUs with IDs: ${GPU_IDS[@]}"

run_counter=0

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
                                for DROPOUTRATE in "${DROPOUTRATES[@]}"; do
                                    for OVERFIT in "${OVERFITS[@]}"; do
                                        for ADAPTIVE in "${ADAPTIVES[@]}"; do
                                          for SOLVER in  "${SOLVERS[@]}"; do
                                             for ONE_WAY in "${ONE_WAYS[@]}"; do
                                              
                                              # --- Compute model scale–dependent parameters ---
                                              this_hidden_size=$(( hidden_size * MODEL_SCALE ))
                                              this_memory_size=$(( memory_size * MODEL_SCALE ))
                                              this_head_size=$(( head_size * MODEL_SCALE ))
                                              # this_num_head=$(( num_heads * MODEL_SCALE ))
                                              this_num_head=$(( num_heads * 1 ))
                                              this_input_size=$(( input_dim * MODEL_SCALE ))
                                              
                                              # --- Loop over new ablation parameters ---
                                              for numPert in "${NUM_PERTURBATIONS[@]}"; do
                                                  
                                                    
                                                    # Determine extra flags based on optimizer type.
                                                    if [[ "$OPTIMIZER_TYPE" == mezo* ]]; then
                                                      BLAH="mezo"
                                                      # FIXED_FLAG="--fixed_size_perturbation"
                                                      EXTRA_FLAGS="--cosine_lr "
                                                      
                                                    else
                                                      BLAH="sgd"
                                                      FIXED_FLAG=""
                                                      EXTRA_FLAGS=""
                                                    fi
            
            
                                                    if [ "$ONE_WAY" = true ]; then
                                                      EXTRA_FLAGS+=" --one_way"
                                                    fi
                                                    
                                                    if [ "$ADAPTIVE" = true ]; then
                                                      EXTRA_FLAGS+=" --adaptive"
                                                    fi
                                                    if [ "$OVERFIT" = true ]; then
                                                      EXTRA_FLAGS+=" --overfit_to_one_batch_flag"
                                                    fi

                                                    EXTRA_FLAGS+=" --cosine_lr"
                                                    
                                                    for agg in "${AGGREGATION_METHODS[@]}"; do
                                                      # Construct run name including model scale parameters.
                                                      # 
                                                      RUN_NAME_BASE="${RUN_PREFIX}_pert_${numPert}_dor_${DROPOUTRATE}_scale${MODEL_SCALE}_OPT${OPTIMIZER_TYPE}_lr${LR}_oneway_${ONE_WAY}_adap_${ADAPTIVE}_overfit_${OVERFIT}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}_npert${numPert}_agg${agg}"

                                                      # --- Assign GPU for this run ---
                                                      gpu_index=$(( run_counter % NUM_GPUS ))
                                                      assigned_gpu_id=${GPU_IDS[$gpu_index]}
                                                      device_string="cuda:${assigned_gpu_id}" # Device string for python script
                                                      # --- END GPU Assignment ---
                                          
                                                      RUN_NAME=$(truncate_name "${RUN_NAME_BASE}")
                                                      echo "[INFO] Launching screen session: $RUN_NAME_BASE"
                                                      
                                                      screen -dmS "$RUN_NAME" bash -c "
                                                        echo '[INFO] Starting run: $RUN_NAME';
                                                        export WANDB_RUN_NAME=$RUN_NAME;
                                                        python ntm_with_modern_training_runs.py \
                                                          --device ${device_string} \
                                                          --task ${TASK} \
                                                          --arch ${ARCH} \
                                                          --minimum_starting_point_context_length ${INPUT_SAMPLE_LENGTH} \
                                                          --hidden_size ${this_hidden_size} \
                                                          --memory_size ${this_memory_size} \
                                                          --head_size ${this_head_size} \
                                                          --num_heads ${this_num_head} \
                                                          --input_size ${this_input_size} \
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
                                                          --probe_dropout_rate ${DROPOUTRATE} \
                                                          --use_same_eps_for_all_perturbations \
                                                          --variance_reduction $variance_reduction \
                                                          --wandb_proj ${WANDB_PROJ} \
                                                          --adam_beta1 ${ADAM_BETA1} \
                                                          --adam_beta2 ${ADAM_BETA2} \
                                                          --solver ${SOLVER} \
                                                          --wandb_run_name ${RUN_NAME_BASE};
                                                        echo '[INFO] Finished run: $RUN_NAME_BASE';
                                                        exec bash
                                                      "
                                                      run_counter=$(( run_counter + 1 ))
                                                      sleep 1
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
            done
          done
        done
      done
    done
  done
done

echo "[INFO] Done launching all screen sessions."
