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
screen -ls | grep '\.NTM_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit # 2>/dev/null

echo "[INFO] Starting to launch screens.."

# 2) Define hyperparameter values
TASKS=("copy" "repeat_copy" "associative_recall" "add" "sub" "mul" "div" "fib" "factorial")
ARCHITECTURES=("ntm" "dnc" "tra")

# You can adjust these as you wish:
HIDDEN_SIZES=(64)
MEMORY_SIZES=(64)
HEAD_SIZES=(32)
LEARNING_RATES=(0.001)
EPSILONS=(0.001)
BATCH_SIZES=(16)
MEZO_LAYERWISE=("false" "true")

OPTIMIZERS=("adam" "mezo")
MAX_ITERS=1000000
SEQ_LEN=10
LOG_INTERVAL=500
WANDB_PROJ="NTM-Experiments"
RUN_PREFIX="NTM_"

# 3) Sweep over each config
for TASK in "${TASKS[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    for HS in "${HIDDEN_SIZES[@]}"; do
      for MEM in "${MEMORY_SIZES[@]}"; do
        for HEAD_SZ in "${HEAD_SIZES[@]}"; do
          for BS in "${BATCH_SIZES[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
              for OPT in "${OPTIMIZERS[@]}"; do

                if [ "$OPT" = "mezo" ]; then
                  # MeZO => we also sweep EPSILON & mezo_layerwise
                  for EPS in "${EPSILONS[@]}"; do
                    for MLW in "${MEZO_LAYERWISE[@]}"; do
                      RUN_NAME="${RUN_PREFIX}${TASK}_${ARCH}_lr${LR}_eps${EPS}_mezo_LW${MLW}"
                      echo "[INFO] Launching screen session: $RUN_NAME"

                      screen -dmS "$RUN_NAME" bash -c "
                        echo '[INFO] Starting run: $RUN_NAME';
                        export WANDB_RUN_NAME=$RUN_NAME;
                        python ntm_with_modern_training_runs.py \
                          --task $TASK \
                          --arch $ARCH \
                          --seq_len $SEQ_LEN \
                          --hidden_size $HS \
                          --memory_size $MEM \
                          --head_size $HEAD_SZ \
                          --num_heads 1 \
                          --batch_size $BS \
                          --max_iters $MAX_ITERS \
                          --log_interval $LOG_INTERVAL \
                          --cosine_lr \
                          --learning_rate $LR \
                          --optimizer mezo \
                          --mezo \
                          --epsilon $EPS \
                          --input_size 32 \
                          $( [ $MLW = 'true' ] && echo '--mezo_layerwise' ) \
                          --wandb_proj $WANDB_PROJ;
                        echo '[INFO] Finished run: $RUN_NAME';
                        exec bash
                      "
                      sleep 1
                    done
                  done

                else
                  # For Adam (standard backprop)
                  RUN_NAME="${RUN_PREFIX}${TASK}_${ARCH}_lr${LR}_OPTadam"
                  echo "[INFO] Launching screen session: $RUN_NAME"

                  screen -dmS "$RUN_NAME" bash -c "
                    echo '[INFO] Starting run: $RUN_NAME';
                    export WANDB_RUN_NAME=$RUN_NAME;
                    python ntm_with_modern_training_runs.py \
                      --task $TASK \
                      --arch $ARCH \
                      --seq_len $SEQ_LEN \
                      --hidden_size $HS \
                      --memory_size $MEM \
                      --head_size $HEAD_SZ \
                      --num_heads 1 \
                      --batch_size $BS \
                      --max_iters $MAX_ITERS \
                      --log_interval $LOG_INTERVAL \
                      --cosine_lr \
                      --learning_rate $LR \
                      --optimizer adam \
                      --input_size 32 \
                      --wandb_proj $WANDB_PROJ;
                    echo '[INFO] Finished run: $RUN_NAME';
                    exec bash
                  "
                  sleep 1
                fi

              done
            done
          done
        done
      done
    done
  done
done

echo "[INFO] Done launching all screen sessions."
