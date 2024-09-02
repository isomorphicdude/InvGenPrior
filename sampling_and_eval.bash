#!/bin/bash

# Run quantiative benchmarks

python run_sampling.py --config configs/tmpd_cg/afhq/inpaint_pixel.py --workdir "Samples"
python run_sampling.py --config configs/tmpd/afhq/inpaint_pixel.py --workdir "Samples"
python run_sampling.py --config configs/pgdm/afhq/inpaint_pixel.py --workdir "Samples"