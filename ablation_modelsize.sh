#!/bin/bash

##############################################################################
# ablation_study_run.sh
#
# Ablation study script for DNC-MICROBATCH-ABLATION
##############################################################################

# 1) Kill existing screens with prefix "NTM_"
echo "[INFO] Killing existing screen sessions named 'NTM_*'..."
screen -ls | grep '\.Run_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting ablation study runs..."

# Define hyperparameter values
TASKS=("owt")
ARCHITECTURES=("simplelstm")
OPTIMIZERS=("mezo_adaptive_sampling_fast" "mezo_single_fast" "anp" )
MODEL_SCALES=(32 128 512 2048 8192)
hidden_size=10
memory_size=10
head_size=10
num_heads=1
input_dim=32
MICRO_BATCH_SIZES=(32)
MACRO_BATCH_SIZES=(1)
INPUT_SAMPLE_LENGTHS=(150)

LEARNING_RATES=(0.001)
EPSILONS=(0.0)
EPS_LR_RATIOS=(0.5)
MAX_NUMS=(120)
WARMUP_STEPS=(100)
WEIGHT_DECAYS=(0)
GRAD_CLIPS=(0)
PAD_BIASES=(0.0)

MAX_ITERS_LIST=(1000)
LOG_INTERVAL=100
WANDB_PROJ="LSTM-MODELSIZE-ABLATION"
RUN_PREFIX="Run_"

# Function to truncate screen session name to avoid overly long names
truncate_name() {
  echo "$1" | cut -c1-65
}

# Sweep over all hyperparameters
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
                              
                              # Adjust MAX_ITERS based on optimizer type
                              for MAX_ITERS in "${MAX_ITERS_LIST[@]}"; do
                                if [ "$OPTIMIZER_TYPE" == "sgd" ]; then
                                  MAX_ITERS=10000
                                  BLAH="sgd"
                                else
                                  BLAH="mezo"
                                fi

                                # Scale model parameters
                                this_hidden_size=$((hidden_size * MODEL_SCALE))
                                this_memory_size=$((memory_size * MODEL_SCALE))
                                this_head_size=$((head_size * MODEL_SCALE))

                                # Construct RUN_NAME
                                RUN_NAME_BASE=$(truncate_name "${RUN_PREFIX}_${ARCH}_${OPTIMIZER_TYPE}_scale${MODEL_SCALE}_iters${MAX_ITERS}_lr${LR}_eps${EPS}_elratio${RATIO}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}")

                                
                                # Check if optimizer is a MeZO optimizer
                                if [[ "$OPTIMIZER_TYPE" == "mezo_adaptive_sampling_fast" || "$OPTIMIZER_TYPE" == "mezo_single_fast"  || "$OPTIMIZER_TYPE" == "anp" ]]; then
                                  # Run with --fixed_size_perturbation
                                  RUN_NAME="${RUN_NAME_BASE}_f"
                                  echo "[INFO] Launching screen session: $RUN_NAME"

                                  screen -dmS "$RUN_NAME" bash -c "
                                    echo '[INFO] Starting run: $RUN_NAME';
                                    export WANDB_RUN_NAME=$RUN_NAME;
                                    python ntm_with_modern_training_runs.py \
                                      --task $TASK \
                                      --arch $ARCH \
                                      --input_sample_length $INPUT_SAMPLE_LENGTH \
                                      --hidden_size $this_hidden_size \
                                      --memory_size $this_memory_size \
                                      --head_size $this_head_size \
                                      --num_heads $num_heads \
                                      --micro_batch_size $MICRO_BS \
                                      --macro_batch_size $MACRO_BS \
                                      --max_iters $MAX_ITERS \
                                      --log_interval $LOG_INTERVAL \
                                      --learning_rate $LR \
                                      --epsilon $EPS \
                                      --tie_epsilon_to_lr_ratio $RATIO \
                                      --max_num $MAX_NUM \
                                      --weight_decay $WEIGHT_DECAY \
                                      --warmup_steps $WARMUP_STEP \
                                      --mezo_flavor $OPTIMIZER_TYPE \
                                      --grad_clip $GRAD_CLIP \
                                      --pad_bias $PAD_BIAS \
                                      --cosine_lr \
                                      --optimizer $BLAH \
                                      --fixed_size_perturbation \
                                      --input_size $input_dim \
                                      --wandb_run_name $RUN_NAME \
                                      --wandb_proj ${WANDB_PROJ};
                                    echo '[INFO] Finished run: $RUN_NAME';
                                    exec bash
                                  "
                                  sleep 4

                                  # Run without --fixed_size_perturbation
                                  RUN_NAME="${RUN_NAME_BASE}_"
                                  echo "[INFO] Launching screen session: $RUN_NAME"

                                  screen -dmS "$RUN_NAME" bash -c "
                                    echo '[INFO] Starting run: $RUN_NAME';
                                    export WANDB_RUN_NAME=$RUN_NAME;
                                    python ntm_with_modern_training_runs.py \
                                      --task $TASK \
                                      --arch $ARCH \
                                      --input_sample_length $INPUT_SAMPLE_LENGTH \
                                      --hidden_size $this_hidden_size \
                                      --memory_size $this_memory_size \
                                      --head_size $this_head_size \
                                      --num_heads $num_heads \
                                      --micro_batch_size $MICRO_BS \
                                      --macro_batch_size $MACRO_BS \
                                      --mezo_flavor $OPTIMIZER_TYPE \
                                      --max_iters $MAX_ITERS \
                                      --log_interval $LOG_INTERVAL \
                                      --learning_rate $LR \
                                      --epsilon $EPS \
                                      --tie_epsilon_to_lr_ratio $RATIO \
                                      --max_num $MAX_NUM \
                                      --weight_decay $WEIGHT_DECAY \
                                      --warmup_steps $WARMUP_STEP \
                                      --grad_clip $GRAD_CLIP \
                                      --pad_bias $PAD_BIAS \
                                      --cosine_lr \
                                      --optimizer $BLAH \
                                      --input_size $input_dim \
                                      --wandb_run_name $RUN_NAME \
                                      --wandb_proj ${WANDB_PROJ};
                                    echo '[INFO] Finished run: $RUN_NAME';
                                    exec bash
                                  "
                                  sleep 4
                                else # handled down below bc separate configs for SGD
                                  # Non-MeZO optimizer run
                                  # RUN_NAME="$RUN_NAME_BASE"
                                  # echo "[INFO] Launching screen session: $RUN_NAME"

                                  # screen -dmS "$RUN_NAME" bash -c "
                                  #   echo '[INFO] Starting run: $RUN_NAME';
                                  #   export WANDB_RUN_NAME=$RUN_NAME;
                                  #   python ntm_with_modern_training_runs.py \
                                  #     --task $TASK \
                                  #     --arch $ARCH \
                                  #     --input_sample_length $INPUT_SAMPLE_LENGTH \
                                  #     --hidden_size $this_hidden_size \
                                  #     --memory_size $this_memory_size \
                                  #     --head_size $this_head_size \
                                  #     --num_heads $num_heads \
                                  #     --micro_batch_size $MICRO_BS \
                                  #     --macro_batch_size $MACRO_BS \
                                  #     --max_iters $MAX_ITERS \
                                  #     --log_interval $LOG_INTERVAL \
                                  #     --learning_rate $LR \
                                  #     --epsilon $EPS \
                                  #     --tie_epsilon_to_lr_ratio $RATIO \
                                  #     --max_num $MAX_NUM \
                                  #     --weight_decay $WEIGHT_DECAY \
                                  #     --warmup_steps $WARMUP_STEP \
                                  #     --grad_clip $GRAD_CLIP \
                                  #     --pad_bias $PAD_BIAS \
                                  #     --cosine_lr \
                                  #     --optimizer $BLAH \
                                  #     --input_size $input_dim \
                                  #     --wandb_run_name $RUN_NAME \
                                  #     --wandb_proj ${WANDB_PROJ};
                                  #   echo '[INFO] Finished run: $RUN_NAME';
                                  #   exec bash
                                  # "
                                  # sleep 4
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

echo "[INFO] Done launching all screen sessions."


############################################
############################################
# # now SGD Runs:
############################################
############################################
#!/bin/bash

# ##############################################################################
# # ablation_study_run.sh
# #
# # Ablation study script for DNC-MICROBATCH-ABLATION
# ##############################################################################

# # 1) Kill existing screens with prefix "NTM_"
# echo "[INFO] Killing existing screen sessions named 'NTM_*'..."
# screen -ls | grep '\.Run_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting ablation study runs..."

# # Define hyperparameter values
TASKS=("owt")
ARCHITECTURES=("simplelstm")
OPTIMIZERS=("sgd")

MODEL_SCALES=(256 64 16 4)
hidden_size=10
memory_size=10
head_size=10
num_heads=1
input_dim=32
MICRO_BATCH_SIZES=(1)
MACRO_BATCH_SIZES=(32)
INPUT_SAMPLE_LENGTHS=(150)

LEARNING_RATES=(0.001)
EPSILONS=(0.0)
EPS_LR_RATIOS=(0.5)
MAX_NUMS=(120)
WARMUP_STEPS=(100)
WEIGHT_DECAYS=(1e-6)
GRAD_CLIPS=(0)
PAD_BIASES=(0.0)

MAX_ITERS_LIST=(10000)
LOG_INTERVAL=100
WANDB_PROJ="LSTM-MODELSIZE-ABLATION"
RUN_PREFIX="Run_"

# Function to truncate screen session name to avoid overly long names
truncate_name() {
  echo "$1" | cut -c1-65
}

# Sweep over all hyperparameters
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
                              
                              # Adjust MAX_ITERS based on optimizer type
                              for MAX_ITERS in "${MAX_ITERS_LIST[@]}"; do
                               
                                # Scale model parameters
                                this_hidden_size=$((hidden_size * MODEL_SCALE))
                                this_memory_size=$((memory_size * MODEL_SCALE))
                                this_head_size=$((head_size * MODEL_SCALE))
                                
                                # Construct RUN_NAME
                                RUN_NAME_BASE=$(truncate_name "${RUN_PREFIX}_${ARCH}_${OPTIMIZER_TYPE}_scale${MODEL_SCALE}_iters${MAX_ITERS}_lr${LR}_eps${EPS}_elratio${RATIO}_cx${INPUT_SAMPLE_LENGTH}_maxnum${MAX_NUM}_hs${this_hidden_size}_mem${this_memory_size}_head${this_head_size}_${TASK}")
                                
                                # Non-MeZO optimizer run
                                RUN_NAME="$RUN_NAME_BASE"
                                echo "[INFO] Launching screen session: $RUN_NAME"
                                
                                screen -dmS "$RUN_NAME" bash -c "
                                echo '[INFO] Starting run: $RUN_NAME';
                                export WANDB_RUN_NAME=$RUN_NAME;
                                python ntm_with_modern_training_runs.py \
                                  --task $TASK \
                                  --arch $ARCH \
                                  --input_sample_length $INPUT_SAMPLE_LENGTH \
                                  --hidden_size $this_hidden_size \
                                  --memory_size $this_memory_size \
                                  --head_size $this_head_size \
                                  --num_heads $num_heads \
                                  --micro_batch_size $MICRO_BS \
                                  --macro_batch_size $MACRO_BS \
                                  --max_iters $MAX_ITERS \
                                  --log_interval $LOG_INTERVAL \
                                  --learning_rate $LR \
                                  --epsilon $EPS \
                                  --tie_epsilon_to_lr_ratio $RATIO \
                                  --max_num $MAX_NUM \
                                  --weight_decay $WEIGHT_DECAY \
                                  --warmup_steps $WARMUP_STEP \
                                  --grad_clip $GRAD_CLIP \
                                  --pad_bias $PAD_BIAS \
                                  --cosine_lr \
                                  --optimizer $OPTIMIZER_TYPE \
                                  --input_size $input_dim \
                                  --wandb_run_name $RUN_NAME \
                                  --wandb_proj ${WANDB_PROJ};
                                echo '[INFO] Finished run: $RUN_NAME';
                                exec bash
                                "
                                sleep 20
                                    

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
