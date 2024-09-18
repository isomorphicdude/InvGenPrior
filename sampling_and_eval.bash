#!/bin/bash

# Run quantiative benchmarks for AFHQ and CelebA datasets

noise_levels=(0.01 0.1 1.0)
# noise_levels=(0.05)
max_num_samples=40
max_num_samples_celeba=1000
nfe=100
gmres_max_iters=(1 2 3 5)


# back up the celeba config
# cp configs/celeb_configs.py configs/celeb_configs_backup.py

for noise_lv in ${noise_levels[@]}; do
    # create a new config by changing the noise level
    echo "Sampling with noise level ${noise_lv}"
    # sed -i "s/sampling\.degredation_sigma = .*/sampling\.degredation_sigma = ${noise_lv}/g" configs/celeb_configs.py

    echo "Running sampling with noise level ${noise_lv}"
    # pixel inpainting
    python run_sampling.py --config configs/tmpd_cg/celeba/inpaint_pixel.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0
    python run_sampling.py --config configs/tmpd_cg/celeba/inpaint_pixel.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0 --tmpd_alt_impl
    # python run_sampling.py --config configs/pgdm/celeba/inpaint_pixel.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0


    # box inpainting
    python run_sampling.py --config configs/tmpd_cg/celeba/inpaint_box.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0
    python run_sampling.py --config configs/tmpd_cg/celeba/inpaint_box.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0 --tmpd_alt_impl
    # python run_sampling.py --config configs/pgdm/celeba/inpaint_box.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0


    # deblurring
    python run_sampling.py --config configs/tmpd_cg/celeba/deblur.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0
    python run_sampling.py --config configs/tmpd_cg/celeba/deblur.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0 --tmpd_alt_impl
    # python run_sampling.py --config configs/tmpd_cgr/celeba/deblur.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0
    # python run_sampling.py --config configs/pgdm/celeba/deblur.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0


    # super-resolution
    python run_sampling.py --config configs/tmpd_cg/celeba/super_res.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0
    python run_sampling.py --config configs/tmpd_cg/celeba/super_res.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0 --tmpd_alt_impl
    # python run_sampling.py --config configs/pgdm/celeba/super_res.py  --max_num_samples ${max_num_samples} --compute_recon_metrics --noise_level ${noise_lv} --starting_time 0.0
done

