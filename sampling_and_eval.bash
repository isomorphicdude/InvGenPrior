#!/bin/bash
# pip install -r requirements.txt

# Run quantiative benchmarks


# sample the images
python run_sampling.py --config configs/tmpd_cg/afhq/inpaint_pixel.py  --max_num_samples 4 --compute_recon_metrics
# python run_sampling.py --config configs/tmpd/afhq/inpaint_pixel.py 
# python run_sampling.py --config configs/pgdm/afhq/inpaint_pixel.py 

# compute metrics
python 
