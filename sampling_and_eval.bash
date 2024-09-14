#!/bin/bash

# Run quantiative benchmarks for AFHQ dataset

# noise_levels=(0.0 0.01 0.05)
noise_levels=(0.05)

# back up the celeba config
cp configs/celeb_configs.py configs/celeb_configs_backup.py

for noise_lv in ${noise_levels[@]}; do
    # create a new config by changing the noise level
    echo "Modifying with noise level ${noise_lv}"
    sed -i "s/sampling\.degredation_sigma = .*/sampling\.degredation_sigma = ${noise_lv}/g" configs/celeb_configs.py

    echo "Running sampling for AFHQ dataset with noise level ${noise_lv}"
    # pixel inpainting
    # python run_sampling.py --config configs/tmpd_cg/afhq/inpaint_pixel.py  --max_num_samples 4 --compute_recon_metrics
    # python run_sampling.py --config configs/tmpd/afhq/inpaint_pixel.py  --max_num_samples 100 --compute_recon_metrics
    # python run_sampling.py --config configs/pgdm/afhq/inpaint_pixel.py  --max_num_samples 4 --compute_recon_metrics
    # python run_sampling.py --config configs/reddiff/afhq/inpaint_pixel.py --max_num_samples 100 --compute_recon_metrics

    # python run_sampling.py --config configs/tmpd_cg/afhq/inpaint_pixel.py --compute_recon_metrics
    # python run_sampling.py --config configs/tmpd/afhq/inpaint_pixel.py --compute_recon_metrics
    #python run_sampling.py --config configs/pgdm/afhq/inpaint_pixel.py --compute_recon_metrics
    #python run_sampling.py --config configs/reddiff/afhq/inpaint_pixel.py --compute_recon_metrics


    # box inpainting
    # python run_sampling.py --config configs/tmpd_cg/afhq/inpaint_box.py  --max_num_samples 100 --compute_recon_metrics
    # python run_sampling.py --config configs/pgdm/afhq/inpaint_box.py  --max_num_samples 100 --compute_recon_metrics
    # python run_sampling.py --config configs/reddiff/afhq/inpaint_box.py --max_num_samples 100 --compute_recon_metrics


    # deblurring
    # python run_sampling.py --config configs/tmpd_cg/afhq/deblur.py  --max_num_samples 20 --compute_recon_metrics
    # python run_sampling.py --config configs/pgdm/afhq/deblur.py  --max_num_samples 4 --compute_recon_metrics
    python run_sampling.py --config configs/reddiff/afhq/deblur.py --max_num_samples 20 --compute_recon_metrics


    # super-resolution
    # python run_sampling.py --config configs/tmpd_cg/afhq/super_res.py  --max_num_samples 100 --compute_recon_metrics
    # python run_sampling.py --config configs/pgdm/afhq/super_res.py  --max_num_samples 100 --compute_recon_metrics
    # python run_sampling.py --config configs/reddiff/afhq/super_res.py --max_num_samples 100 --compute_recon_metrics
done

# restore the celeba config
mv configs/celeb_configs_backup.py configs/celeb_configs.py

