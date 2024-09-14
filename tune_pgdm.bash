#!/bin/bash

noise_levels=(0.0 0.01 0.05 0.1)
starting_times=(0.0 0.05 0.1 0.2 0.3)

# List of configuration files to run
configs=(
    "configs/pgdm/afhq/inpaint_pixel.py"
    "configs/pgdm/afhq/inpaint_box.py"
    "configs/pgdm/afhq/deblur.py"
    "configs/pgdm/afhq/super_res.py"
)

# Calculate total number of tasks
total_noise=${#noise_levels[@]}
total_starting=${#starting_times[@]}
total_configs=${#configs[@]}
total_tasks=$((total_noise * total_starting * total_configs))

echo "Starting hyperparameter tuning for PGDM"
echo "Total tasks to run: $total_tasks"
echo "---------------------------------------------"

# Initialize counters
task_count=0

# Iterate over noise levels
for noise_lv in "${noise_levels[@]}"; do
    echo "Tuning with Noise Level: ${noise_lv}"
    
    # Iterate over starting times
    for starting_time in "${starting_times[@]}"; do
        echo "  Starting Time: ${starting_time}"
        
        # Iterate over configuration files
        for config in "${configs[@]}"; do
            task_count=$((task_count + 1))
            tasks_left=$((total_tasks - task_count))
            echo "    [Task $task_count/$total_tasks] Running: $config"
            
            # Execute the Python script
            python run_sampling.py --config "$config" \
                --max_num_samples 4s \
                --compute_recon_metrics \
                --noise_level "$noise_lv" \
                --starting_time "$starting_time"
            
            echo "    [Task $task_count/$total_tasks] Completed. Tasks remaining: $tasks_left"
            echo ""
        done
    done
done

echo "---------------------------------------------"
echo "All tasks completed."
echo "End Time: $(date)"