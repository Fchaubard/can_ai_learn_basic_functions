#!/bin/bash

##############################################################################
# run_sweep_screens.sh
#
# Example shell script to:
# 1) Kill existing screen sessions that start with "NTM_"
# 2) Launch new screen sessions for each hyperparam combination
#
# Usage:
#   chmod +x run_sweep_screens.sh
#   ./run_sweep_screens.sh
##############################################################################

# 1) Kill existing screens with prefix "NTM_"
echo "[INFO] Killing existing screen sessions named 'NTM_*'..."
screen -ls | grep '\.NTM_' | awk '{print $1}' | cut -c1-80 | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting to launch screens.."

# 2) Define hyperparameter values
TASKS=("add")
# TASKS=("copy" "repeat_copy" "associative_recall" "add" "sub" "mul" "div" "fib" "factorial")


# ARCHITECTURES=("ntm" "dnc" "tra")
ARCHITECTURES=("ntm") 


HIDDEN_SIZES=(128)
MEMORY_SIZES=(128)
HEAD_SIZES=(64)
LEARNING_RATES=(0.001 0.0001 0.00001 0.000001)
EPSILONS=(0.01 0.001 0.0001 0.00001)

MICRO_BATCH_SIZES=(32)
MACRO_BATCH_SIZES=(1)
CONTEXT_LENGTHS=(10)
MAX_SEQ_LENS=(50)

GRAD_CLIPS=(0 1.0)
PAD_BIASES=(0.0 2.0)

MEZO_LAYERWISE=("false" "true")
OPTIMIZERS=("adam" "mezo")

MAX_ITERS=100000
LOG_INTERVAL=500
WANDB_PROJ="NTM-Experiments"
RUN_PREFIX="NTM_"

# Function to truncate names to 80 characters
truncate_name() {
  echo "$1" | cut -c1-72
}

# 3) Sweep
for TASK in "${TASKS[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    for HS in "${HIDDEN_SIZES[@]}"; do
      for MEM in "${MEMORY_SIZES[@]}"; do
        for HEAD_SZ in "${HEAD_SIZES[@]}"; do
          for LR in "${LEARNING_RATES[@]}"; do
            for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
              for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
                for CXLEN in "${CONTEXT_LENGTHS[@]}"; do
                  for MSLEN in "${MAX_SEQ_LENS[@]}"; do
                    for GC in "${GRAD_CLIPS[@]}"; do
                      for PAD_B in "${PAD_BIASES[@]}"; do
                        for OPT in "${OPTIMIZERS[@]}"; do

                          if [ "$OPT" = "mezo" ]; then
                            for EPS in "${EPSILONS[@]}"; do
                              for MLW in "${MEZO_LAYERWISE[@]}"; do
                                if [ "$MLW" = "true" ]; then
                                  MLW_FLAG="--mezo_layerwise"
                                else
                                  MLW_FLAG=""
                                fi

                                RUN_NAME=$(truncate_name "${RUN_PREFIX}${TASK}_${ARCH}_lr${LR}_eps${EPS}_mezo_LW${MLW}_gc${GC}_mb${MICRO_BS}x${MACRO_BS}_cx${CXLEN}_maxl${MSLEN}_padb${PAD_B}")
                                echo "[INFO] Launching screen session: $RUN_NAME"

                                screen -dmS "$RUN_NAME" bash -c "
                                  echo '[INFO] Starting run: $RUN_NAME';
                                  export WANDB_RUN_NAME=$RUN_NAME;
                                  python ntm_with_modern_training_runs.py \
                                    --task $TASK \
                                    --arch $ARCH \
                                    --context_length $CXLEN \
                                    --max_seq_len $MSLEN \
                                    --hidden_size $HS \
                                    --memory_size $MEM \
                                    --head_size $HEAD_SZ \
                                    --num_heads 1 \
                                    --micro_batch_size $MICRO_BS \
                                    --macro_batch_size $MACRO_BS \
                                    --max_iters $MAX_ITERS \
                                    --log_interval $LOG_INTERVAL \
                                    --cosine_lr \
                                    --learning_rate $LR \
                                    --optimizer mezo \
                                    --mezo \
                                    $MLW_FLAG \
                                    --epsilon $EPS \
                                    --input_size 32 \
                                    --grad_clip $GC \
                                    --pad_bias $PAD_B \
                                    --wandb_proj $WANDB_PROJ;
                                  echo '[INFO] Finished run: $RUN_NAME';
                                  exec bash
                                "
                                sleep 5
                              done
                            done
                          else
                            RUN_NAME=$(truncate_name "${RUN_PREFIX}${TASK}_${ARCH}_lr${LR}_OPTadam_gc${GC}_mb${MICRO_BS}x${MACRO_BS}_cx${CXLEN}_maxl${MSLEN}_padb${PAD_B}")
                            echo "[INFO] Launching screen session: $RUN_NAME"

                            screen -dmS "$RUN_NAME" bash -c "
                              echo '[INFO] Starting run: $RUN_NAME';
                              export WANDB_RUN_NAME=$RUN_NAME;
                              python ntm_with_modern_training_runs.py \
                                --task $TASK \
                                --arch $ARCH \
                                --context_length $CXLEN \
                                --max_seq_len $MSLEN \
                                --hidden_size $HS \
                                --memory_size $MEM \
                                --head_size $HEAD_SZ \
                                --num_heads 1 \
                                --micro_batch_size $MICRO_BS \
                                --macro_batch_size $MACRO_BS \
                                --max_iters $MAX_ITERS \
                                --log_interval $LOG_INTERVAL \
                                --cosine_lr \
                                --learning_rate $LR \
                                --optimizer adam \
                                --input_size 32 \
                                --grad_clip $GC \
                                --pad_bias $PAD_B \
                                --wandb_proj $WANDB_PROJ;
                              echo '[INFO] Finished run: $RUN_NAME';
                              exec bash
                            "
                            sleep 5
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

echo "[INFO] Done launching all screen sessions."
