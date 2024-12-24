#!/bin/bash

##############################################################################
# run_sweep_screens.sh
#
# Example shell script to:
# 1) Kill existing screen sessions that start with "NTM_"
# 2) Launch new screen sessions for each hyperparam combination
#
# Usage:
#   chmod +x ntm_sweep.sh
#   ./ntm_sweep.sh
##############################################################################

# 1) Kill existing screens with prefix "NTM_"
echo "[INFO] Killing existing screen sessions named 'NTM_*'..."
screen -ls | grep '\.NTM_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

# 2) Hyperparameter values to sweep
TASKS=("copy" "repeat_copy" "associative_recall")
HIDDEN_SIZES=(64 128)
MEMORY_SIZES=(64 128)
HEAD_SIZES=(32 64)
LEARNING_RATES=(0.001 0.0001)
OPTIMIZERS=("adam" "mezo")

MAX_ITERS=5000
BATCH_SIZE=16
SEQ_LEN=10
LOG_INTERVAL=500

WANDB_PROJ="NTM-Experiments"
RUN_PREFIX="NTM_"

# 3) Sweep over each config
for TASK in "${TASKS[@]}"; do
  for HS in "${HIDDEN_SIZES[@]}"; do
    for MEM in "${MEMORY_SIZES[@]}"; do
      for HEAD_SZ in "${HEAD_SIZES[@]}"; do
        for LR in "${LEARNING_RATES[@]}"; do
          for OPT in "${OPTIMIZERS[@]}"; do

            if [ "$OPT" = "mezo" ]; then
              # For MeZO, we can optionally sweep mezo_layerwise as well
              for MLW in "false" "true"; do
                RUN_NAME="${RUN_PREFIX}${TASK}_hs${HS}_mem${MEM}_head${HEAD_SZ}_lr${LR}_OPTmezo_LW${MLW}"
                echo "[INFO] Launching screen session: $RUN_NAME"

                # Launch the run in a new screen session
                screen -dmS "$RUN_NAME" bash -c "
                  echo '[INFO] Starting run: $RUN_NAME';
                  export WANDB_RUN_NAME=$RUN_NAME;
                  python ntm_with_modern_training_runs.py \
                    --task $TASK \
                    --seq_len $SEQ_LEN \
                    --hidden_size $HS \
                    --memory_size $MEM \
                    --cosine_lr \
                    --head_size $HEAD_SZ \
                    --num_heads 1 \
                    --batch_size $BATCH_SIZE \
                    --max_iters $MAX_ITERS \
                    --log_interval $LOG_INTERVAL \
                    --learning_rate $LR \
                    --optimizer mezo \
                    --mezo \
                    $( [ $MLW = 'true' ] && echo '--mezo_layerwise' ) \
                    --wandb_proj $WANDB_PROJ;
                  echo '[INFO] Finished run: $RUN_NAME';
                  exec bash
                "

                # Optional short sleep to let resources settle
                sleep 5
              done
            else
              # For Adam (standard backprop)
              RUN_NAME="${RUN_PREFIX}${TASK}_hs${HS}_mem${MEM}_head${HEAD_SZ}_lr${LR}_OPTadam"
              echo "[INFO] Launching screen session: $RUN_NAME"

              screen -dmS "$RUN_NAME" bash -c "
                echo '[INFO] Starting run: $RUN_NAME';
                export WANDB_RUN_NAME=$RUN_NAME;
                python train_ntm.py \
                  --task $TASK \
                  --seq_len $SEQ_LEN \
                  --hidden_size $HS \
                  --memory_size $MEM \
                  --head_size $HEAD_SZ \
                  --num_heads 1 \
                  --cosine_lr \
                  --batch_size $BATCH_SIZE \
                  --max_iters $MAX_ITERS \
                  --log_interval $LOG_INTERVAL \
                  --learning_rate $LR \
                  --optimizer adam \
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

echo "[INFO] Done launching all screen sessions."
