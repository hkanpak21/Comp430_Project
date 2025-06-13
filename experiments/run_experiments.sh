#!/bin/bash

# Create output directory
OUTPUT_DIR="experiments/out/colab_runs"
mkdir -p $OUTPUT_DIR

# Date stamp for run IDs
DATE_STAMP=$(date +"%Y%m%d_%H%M%S")

# List of experiments to run
declare -a experiments=(
    # Standard comparison - Vanilla vs Adaptive DP
    "configs/mnist_clients5_vanilla_dp.yaml:mnist_clients5_vanilla"
    "configs/mnist_clients5_adaptive_dp.yaml:mnist_clients5_adaptive"
    
    # Extreme non-IID comparison - Vanilla vs Adaptive DP
    "configs/mnist_clients_extreme_noniid_vanilla_dp.yaml:mnist_noniid_vanilla"
    "configs/mnist_clients_extreme_noniid_adaptive_dp.yaml:mnist_noniid_adaptive"
    
    # FL vs SFL comparison - Vanilla DP
    "configs/mnist_fl_vs_sfl_vanilla.yaml:mnist_fl_vanilla"
    "configs/mnist_sfl_vs_fl_vanilla.yaml:mnist_sfl_vanilla"
    
    # FL vs SFL comparison - Adaptive DP
    "configs/mnist_fl_vs_sfl_adaptive.yaml:mnist_fl_adaptive"
    "configs/mnist_sfl_vs_fl_adaptive.yaml:mnist_sfl_adaptive"
    
    # BCW dataset comparison
    "configs/bcw_clients5_vanilla_dp.yaml:bcw_vanilla"
    "configs/bcw_clients5_adaptive_dp.yaml:bcw_adaptive"
)

# Total number of experiments
NUM_EXPERIMENTS=${#experiments[@]}
echo "Running $NUM_EXPERIMENTS experiments"

# Store summary data
summary_file="$OUTPUT_DIR/experiment_summary.json"
echo "[" > $summary_file

# Run each experiment
experiment_num=1
for exp in "${experiments[@]}"; do
    # Split config file and run ID
    IFS=':' read -r config_file run_id <<< "$exp"
    run_id="${run_id}_${DATE_STAMP}"
    
    echo ""
    echo "Running experiment $experiment_num/$NUM_EXPERIMENTS"
    echo ""
    echo "================================================================================"
    echo "Starting experiment: $(basename ${config_file%.*})"
    echo "Config file: $config_file"
    echo "Run ID: $run_id"
    echo "================================================================================"
    echo ""
    
    # Record start time
    start_time=$(date +%s)
    
    # Run the experiment
    cmd="python experiments/train_secure_sfl.py --config $config_file --run_id $run_id"
    echo "Command: $cmd"
    eval $cmd
    
    # Record end time and calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    duration_mins=$(echo "scale=2; $duration / 60" | bc)
    
    echo ""
    echo "Experiment completed in $duration seconds ($duration_mins minutes)"
    
    # Add to summary file
    if [ $experiment_num -lt $NUM_EXPERIMENTS ]; then
        echo "  {\"config\": \"$config_file\", \"run_id\": \"$run_id\", \"duration_seconds\": $duration}," >> $summary_file
    else
        echo "  {\"config\": \"$config_file\", \"run_id\": \"$run_id\", \"duration_seconds\": $duration}" >> $summary_file
    fi
    
    # Increment counter
    ((experiment_num++))
done

# Close summary file
echo "]" >> $summary_file

echo "All $NUM_EXPERIMENTS experiments completed"
echo "Summary saved to $summary_file" 