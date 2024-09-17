#!/bin/bash


noise_levels=(0.0 0.1 1.0)
# noise_levels=(0.05)

gmres_max_iters=(1 2 3 5)
max_samp=20

for noise_lv in ${noise_levels[@]}; do
    echo "Tuning hyperparameters for TMPD with noise level ${noise_lv}"

    for max_iter in ${gmres_max_iters[@]}; do
        python run_sampling.py --config configs/tmpd_cg/afhq/deblur.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
        python run_sampling.py --config configs/tmpd_gmres/afhq/deblur.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
        python run_sampling.py --config configs/tmpd_cgr/afhq/deblur.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
    done
done

# find best hyperparameters
# python evaluation/find_hyp.py --method tmpd --data_name afhq