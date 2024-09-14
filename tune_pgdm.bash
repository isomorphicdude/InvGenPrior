#!/bin/bash

# Tune hyperparameters for PGDM
# only one hyperparameter: starting_time

noise_levels=(0.0 0.01 0.05 0.1)
starting_times=(0.0 0.05 0.1 0.2 0.3)

for noise_lv in ${noise_levels[@]}; do
    echo "Tuning hyperparameters for PGDM with noise level ${noise_lv}"

    for starting_time in ${starting_times[@]}; do
        echo "Sampling with starting time ${starting_time}"
        python run_sampling.py --config configs/pgdm/afhq/inpaint_pixel.py  --max_num_samples 100 --compute_recon_metrics --noise_level ${noise_lv} --starting_time ${starting_time}
        python run_sampling.py --config configs/pgdm/afhq/inpaint_box.py  --max_num_samples 100 --compute_recon_metrics --noise_level ${noise_lv} --starting_time ${starting_time}
        python run_sampling.py --config configs/pgdm/afhq/deblur.py  --max_num_samples 100 --compute_recon_metrics --noise_level ${noise_lv} --starting_time ${starting_time}
        python run_sampling.py --config configs/pgdm/afhq/super_res.py  --max_num_samples 100 --compute_recon_metrics --noise_level ${noise_lv} --starting_time ${starting_time}
    done

    


    