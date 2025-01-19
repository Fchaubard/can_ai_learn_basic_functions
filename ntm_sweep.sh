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
# 
# Example:
# python ntm_with_modern_training_runs.py \
#     --task copy \
#     --arch simplelstm \
#     --input_sample_length 20 \
#     --max_seq_len 12 \
#     --hidden_size 1024 \
#     --memory_size 1024 \
#     --max_num 150 \
#     --head_size 256 \
#     --num_heads 1 \
#     --tie_epsilon_to_lr_ratio .1 \
#     --micro_batch_size 32 \
#     --macro_batch_size 1 \
#     --max_iters 10000 \
#     --log_interval 500 \
#     --cosine_lr \
#     --learning_rate 0.001 \
#     --optimizer mezo \
#     --mezo_flavor mezo_single \
#     --epsilon 0 \
#     --input_size 32 \
#     --grad_clip 0 \
#     --pad_bias 0 \
#     --weight_decay 1e-7 \
#     --wandb_proj NTM-Experiments
##############################################################################
# 1 single lstm 1 (running, result=0.8 loss )
# 2 single lstm 2 (running, result=1.3 loss )
# 3 single lstm with 2 and .1th the eps and lr (running, result=0.72 loss )
# 4 layerwise (running, result=2 )
# 5 ntm sgd at 5 (running, result=solve by 400 iters! )
# 6 ntm single  (running, result=0.7 loss )
# 7 ntm layerwise  (running, result=1.8 )
# 8 tra sgd  (running, result=solve by 400 iters! )
# 9 dnc sgd at 5 (running, result=solve by 400 iters! )
# 10 dnc single  (running, result=0.8 loss )
# 11 dnc layerwise  (running, result=1.3 loss )

# Results:all architectures do work, simplelstm, ntm, tra, and dnc with sgd. 

# 1) Kill existing screens with prefix "NTM_"
echo "[INFO] Killing existing screen sessions named 'NTM_*'..."
screen -ls | grep '\.Run_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

echo "[INFO] Starting to launch screens.."

# 2) Define hyperparameter values (adjust as needed)
TASKS=("copy")
ARCHITECTURES=("simplelstm")

HIDDEN_SIZES=(1024)
MEMORY_SIZES=(1024)
HEAD_SIZES=(4)

LEARNING_RATES=(0.01 0.001 0.0001 0.00001)
EPSILONS=(0) #0.01 0.001 0.0001 0.00001)

MICRO_BATCH_SIZES=(32)
MACRO_BATCH_SIZES=(1)
INPUT_SAMPLE_LENGTHS=(1)
MAX_SEQ_LENS=(15)

GRAD_CLIPS=(0 .5)
PAD_BIASES=(0.0)

OPTIMIZERS=("sgd" "mezo")

MAX_NUMS=(120)                # from --max_num
WARMUP_STEPS=(100)             # from --warmup_steps
WEIGHT_DECAYS=(1e-5 0.0)         # from --weight_decay
MEZO_FLAVORS=("mezo_single" "mezo_layerwise")
EPS_LR_RATIOS=(0.1,0.5, 1.0, 5.0, 10.0) 
MAX_ITERS=100000
LOG_INTERVAL=500
WANDB_PROJ="Final_Experiments"
RUN_PREFIX="Run_"

# Function to truncate screen session name to avoid overly long names
truncate_name() {
  echo "$1" | cut -c1-72
}

# 3) Sweep over all hyperparameters
for TASK in "${TASKS[@]}"; do
  for ARCH in "${ARCHITECTURES[@]}"; do
    for HS in "${HIDDEN_SIZES[@]}"; do
      for MEM in "${MEMORY_SIZES[@]}"; do
        for HEAD_SZ in "${HEAD_SIZES[@]}"; do
          for LR in "${LEARNING_RATES[@]}"; do
            for MICRO_BS in "${MICRO_BATCH_SIZES[@]}"; do
              for MACRO_BS in "${MACRO_BATCH_SIZES[@]}"; do
                for CXLEN in "${INPUT_SAMPLE_LENGTHS[@]}"; do
                  for MSLEN in "${MAX_SEQ_LENS[@]}"; do
                    for GC in "${GRAD_CLIPS[@]}"; do
                      for PAD_B in "${PAD_BIASES[@]}"; do

                        # NEW loops for extra args
                        for MN in "${MAX_NUMS[@]}"; do
                          for WS in "${WARMUP_STEPS[@]}"; do
                            for WD in "${WEIGHT_DECAYS[@]}"; do

                              for OPT in "${OPTIMIZERS[@]}"; do
                                if [ "$OPT" = "mezo" ]; then
                                  for EPS in "${EPSILONS[@]}"; do
                                    # We also loop over the mezo_flavor
                                    for MFLAV in "${MEZO_FLAVORS[@]}"; do
                                        for RATIO in "${EPS_LR_RATIOS[@]}"; do
    
                                          # If user sets --mezo_flavor=None, we skip passing it. Otherwise pass it.
                                          if [ "$MFLAV" != "None" ]; then
                                            MEZO_FLAV_FLAG="--mezo_flavor $MFLAV"
                                          else
                                            MEZO_FLAV_FLAG=""
                                          fi
    
                                          RUN_NAME=$(truncate_name "${RUN_PREFIX}${TASK}_${ARCH}_lr${LR}_eps${EPS}_flav${MFLAV}_gc${GC}_mb${MICRO_BS}x${MACRO_BS}_cx${CXLEN}_maxl${MSLEN}_padb${PAD_B}_maxNum${MN}_wd${WD}_ws${WS}")
                                          echo "[INFO] Launching screen session: $RUN_NAME"
    
                                          screen -dmS "$RUN_NAME" bash -c "
                                            echo '[INFO] Starting run: $RUN_NAME';
                                            export WANDB_RUN_NAME=$RUN_NAME;
                                            python ntm_with_modern_training_runs.py \
                                              --task $TASK \
                                              --arch $ARCH \
                                              --input_sample_length $CXLEN \
                                              --max_seq_len $MSLEN \
                                              --max_num $MN \
                                              --hidden_size $HS \
                                              --memory_size $MEM \
                                              --head_size $HEAD_SZ \
                                              --num_heads 1 \
                                              -tie_epsilon_to_lr_ratio $RATIO \
                                              --micro_batch_size $MICRO_BS \
                                              --macro_batch_size $MACRO_BS \
                                              --max_iters $MAX_ITERS \
                                              --log_interval $LOG_INTERVAL \
                                              --cosine_lr \
                                              --warmup_steps $WS \
                                              --learning_rate $LR \
                                              --weight_decay $WD \
                                              --optimizer mezo \
                                              --mezo \
                                              $MEZO_FLAV_FLAG \
                                              --epsilon $EPS \
                                              --input_size 32 \
                                              --grad_clip $GC \
                                              --pad_bias $PAD_B \
                                              --wandb_proj {$WANDB_PROJ}_{$ARCH}_{$TASK};
                                            echo '[INFO] Finished run: $RUN_NAME';
                                            exec bash
                                          "
                                          sleep 5

                                        done
                                    done
                                  done
                                else
                                  # OPT = sgd
                                  RUN_NAME=$(truncate_name "${RUN_PREFIX}${TASK}_${ARCH}_lr${LR}_OPTsgd_gc${GC}_mb${MICRO_BS}x${MACRO_BS}_cx${CXLEN}_maxl${MSLEN}_padb${PAD_B}_maxNum${MN}_wd${WD}_ws${WS}")
                                  echo "[INFO] Launching screen session: $RUN_NAME"

                                  screen -dmS "$RUN_NAME" bash -c "
                                    echo '[INFO] Starting run: $RUN_NAME';
                                    export WANDB_RUN_NAME=$RUN_NAME;
                                    python ntm_with_modern_training_runs.py \
                                      --task $TASK \
                                      --arch $ARCH \
                                      --input_sample_length $CXLEN \
                                      --max_seq_len $MSLEN \
                                      --max_num $MN \
                                      --hidden_size $HS \
                                      --memory_size $MEM \
                                      --head_size $HEAD_SZ \
                                      --num_heads 1 \
                                      --micro_batch_size $MICRO_BS \
                                      --macro_batch_size $MACRO_BS \
                                      --max_iters $MAX_ITERS \
                                      --log_interval $LOG_INTERVAL \
                                      --cosine_lr \
                                      --warmup_steps $WS \
                                      --learning_rate $LR \
                                      --weight_decay $WD \
                                      --optimizer sgd \
                                      --input_size 32 \
                                      --grad_clip $GC \
                                      --pad_bias $PAD_B \
                                      --wandb_proj {$WANDB_PROJ}_{$ARCH}_{$TASK};
                                    echo '[INFO] Finished run: $RUN_NAME';
                                    exec bash
                                  "
                                  sleep 5
                                fi

                              done # end OPT
                            done   # end WD
                          done     # end WS
                        done       # end MN

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
