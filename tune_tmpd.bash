#!/bin/bash


noise_levels_afhq=(0.01)
noise_levels_celeba=(0.1)
# noise_levels=(0.05)

gmres_max_iters=(1 2)
max_samp=4

for noise_lv in ${noise_levels_afhq[@]}; do
    echo "Tuning hyperparameters for TMPD with noise level ${noise_lv}"

    for max_iter in ${gmres_max_iters[@]}; do
        python run_sampling.py --config configs/tmpd_cg/afhq/deblur.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
        python run_sampling.py --config configs/tmpd_cg/afhq/deblur.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter} --tmpd_alt_impl
        python run_sampling.py --config configs/pgdm/afhq/deblur.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv}
        # python run_sampling.py --config configs/tmpd_gmres/afhq/inpaint_box.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
        # python run_sampling.py --config configs/tmpd_cgr/afhq/inpaint_box.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
    done
    
    # for max_iter in ${gmres_max_iters[@]}; do
    #     python run_sampling.py --config configs/tmpd_cg/afhq/super_res.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
    #     python run_sampling.py --config configs/tmpd_gmres/afhq/super_res.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
    #     python run_sampling.py --config configs/tmpd_cgr/afhq/super_res.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
    # done
done

# for noise_lv in ${noise_levels_celeba[@]}; do
#     echo "Tuning hyperparameters for TMPD with noise level ${noise_lv}"

#     for max_iter in ${gmres_max_iters[@]}; do
#         python run_sampling.py --config configs/tmpd_cg/celeba/inpaint_box.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
#         python run_sampling.py --config configs/tmpd_gmres/celeba/inpaint_box.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
#         python run_sampling.py --config configs/tmpd_cgr/celeba/inpaint_box.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
#     done


#     for max_iter in ${gmres_max_iters[@]}; do
#         python run_sampling.py --config configs/tmpd_cg/celeba/super_res.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
#         python run_sampling.py --config configs/tmpd_gmres/celeba/super_res.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
#         python run_sampling.py --config configs/tmpd_cgr/celeba/super_res.py  --max_num_samples ${max_samp} --compute_recon_metrics --noise_level ${noise_lv} --gmres_max_iter ${max_iter}
#     done
# done

# find best hyperparameters
# python evaluation/find_hyp.py --method tmpd --data_name afhq