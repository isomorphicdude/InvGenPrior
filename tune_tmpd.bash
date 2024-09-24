#!/bin/bash

noise_levels_afhq=(0.01)
noise_levels_celeba=(0.01)
gmres_max_iters=(2)
max_samp=100
nfe=100

# Function to calculate and display the time in a readable format
display_time() {
    local T=$1
    local D=$((T/60/60/24))
    local H=$((T/60/60%24))
    local M=$((T/60%60))
    local S=$((T%60))
    if [ ${D} -gt 0 ]; then
        printf '%d days %d hours %d minutes %d seconds\n' $D $H $M $S
    elif [ ${H} -gt 0 ]; then
        printf '%d hours %d minutes %d seconds\n' $H $M $S
    elif [ ${M} -gt 0 ]; then
        printf '%d minutes %d seconds\n' $M $S
    else
        printf '%d seconds\n' $S
    fi
}

# Track total number of experiments to estimate time remaining
total_experiments=$((3 * (${#noise_levels_afhq[@]} + ${#noise_levels_celeba[@]}) * ${#gmres_max_iters[@]} * 4))

echo "============================================"
echo "Total number of experiments: ${total_experiments}"
echo "current_time: $(date)"
echo "============================================"

experiment_count=0
start_time=$(date +%s)

for noise_lv in ${noise_levels_afhq[@]}; do
    echo "Tuning hyperparameters for TMPD with noise level ${noise_lv}"

    # AFHQ
    for max_iter in ${gmres_max_iters[@]}; do
        for task in "deblur" "inpaint_box" "inpaint_pixel" "super_res"; do
            python run_sampling.py --config configs/tmpd_cg/afhq/${task}.py --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --nfe ${nfe}
            python run_sampling.py --config configs/tmpd_gmres/afhq/${task}.py --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --nfe ${nfe}
            python run_sampling.py --config configs/tmpd_cgr/afhq/${task}.py --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --nfe ${nfe}
            
            # Update experiment count and show elapsed/remaining time
            ((experiment_count+=3))
            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time))
            average_time=$((elapsed_time / experiment_count))
            remaining_time=$((average_time * (total_experiments - experiment_count)))

            echo "============================================"
            echo "Completed experiment ${experiment_count}/${total_experiments}"
            echo -n "Elapsed time: "
            display_time $elapsed_time
            echo -n "Estimated remaining time: "
            display_time $remaining_time
            echo "============================================"
        done
    done
done

for noise_lv in ${noise_levels_celeba[@]}; do
    echo "Tuning hyperparameters for TMPD with noise level ${noise_lv}"

    # CELEBA
    for max_iter in ${gmres_max_iters[@]}; do
        for task in "deblur" "inpaint_box" "inpaint_pixel" "super_res"; do
            python run_sampling.py --config configs/tmpd_cg/celeba/${task}.py --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --nfe ${nfe}
            python run_sampling.py --config configs/tmpd_gmres/celeba/${task}.py --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --nfe ${nfe}
            python run_sampling.py --config configs/tmpd_cgr/celeba/${task}.py --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --nfe ${nfe}
            
            # Update experiment count and show elapsed/remaining time
            ((experiment_count+=3))
            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time))
            average_time=$((elapsed_time / experiment_count))
            remaining_time=$((average_time * (total_experiments - experiment_count)))
            echo "============================================"
            echo "Completed experiment ${experiment_count}/${total_experiments}"
            echo -n "Elapsed time: "
            display_time $elapsed_time
            echo -n "Estimated remaining time: "
            display_time $remaining_time
            echo "============================================"
        done
    done
done