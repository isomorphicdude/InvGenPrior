#!/bin/bash
# pip install -r requirements.txt

# Run quantiative benchmarks for AFHQ dataset

# pixel inpainting
# python run_sampling.py --config configs/tmpd_cg/afhq/inpaint_pixel.py  --max_num_samples 100 --compute_recon_metrics
# python run_sampling.py --config configs/tmpd/afhq/inpaint_pixel.py  --max_num_samples 100 --compute_recon_metrics
python run_sampling.py --config configs/pgdm/afhq/inpaint_pixel.py  --max_num_samples 100 --compute_recon_metrics
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
# python run_sampling.py --config configs/tmpd_cg/afhq/deblur.py  --max_num_samples 100 --compute_recon_metrics
# python run_sampling.py --config configs/pgdm/afhq/deblur.py  --max_num_samples 100 --compute_recon_metrics
# python run_sampling.py --config configs/reddiff/afhq/deblur.py --max_num_samples 100 --compute_recon_metrics


# super-resolution
# python run_sampling.py --config configs/tmpd_cg/afhq/super_res.py  --max_num_samples 100 --compute_recon_metrics
# python run_sampling.py --config configs/pgdm/afhq/super_res.py  --max_num_samples 100 --compute_recon_metrics
# python run_sampling.py --config configs/reddiff/afhq/super_res.py --max_num_samples 100 --compute_recon_metrics


