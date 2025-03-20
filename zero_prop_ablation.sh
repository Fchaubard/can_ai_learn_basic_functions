#!/bin/bash
echo "[INFO] Killing existing screen sessions named 'NTM_*'..."
screen -ls | grep '\.dnc_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null

# Base master port for training jobs
base_port=29501

meta_perturbations=1

# Define arrays for probe dropout rates and learning rates
DOR=(0.99 0.999)
LR=(0.01 0.001)  # Example values; adjust as needed

# Loop over each dropout rate and learning rate combination
for dor in "${DOR[@]}"; do
  for lr in "${LR[@]}"; do
    ((base_port++))  # Increment the base port
    port=${base_port}
    session_name="dnc_${port}"
    echo "[INFO] Launching training job on port ${port} in session ${session_name}..."
    
    screen -dmS "${session_name}" bash -c "python -m torch.distributed.launch --nproc_per_node=10 --master_port=${port} dnc_distributed.py --meta_perturbations ${meta_perturbations} --probe_dropout_rate ${dor} --learning_rate ${lr}; exec bash"
    
    sleep 2  # Optional: wait a bit before launching the next job
  done
done

echo "[INFO] All training jobs launched."
