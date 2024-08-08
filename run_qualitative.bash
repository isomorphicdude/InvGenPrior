#!/bin/bash

# Run the qualitative evaluation
# sigma 0.0, 0.01, 0.05
# sample_N 50, 100
# inpaint box 52
# inpaint pixel 53
# deblur 2
# super-res 15

sigma_list=(0.0 0.01 0.05)
# sigma_list=(0.05)

for sigma in ${sigma_list[@]}; do
    echo "Running qualitative evaluation for sigma = $sigma for AFHQ dataset"
    # inpaint box
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 322
    # inpaint pixel
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 53
    # deblur
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/deblur.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 2
    # super-res
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/super_res.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 15
done


for sigma in ${sigma_list[@]}; do
    echo "Running qualitative evaluation for sigma = $sigma for CelebA dataset"
    # inpaint box
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 322
    # inpaint pixel
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 53
    # deblur
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/deblur.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 2
    # super-res
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/super_res.py --workdir /content/InvGenPrior/ --sample_N 100 --noise_sigma $sigma --data_index 15
done
