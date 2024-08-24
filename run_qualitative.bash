#!/bin/bash

# Run the qualitative evaluation
# sigma 0.0, 0.01, 0.05
# sample_N 50, 100
# inpaint box 52
# inpaint pixel 53
# deblur 2
# super-res 15

# sigma_list=(0.0 0.01 0.05)
sigma_list=(0.01)
sample_N=100
sample_count=10

afhq_min_index=0
afhq_max_index=499

celeba_min_index=0
celeba_max_index=1999

for ((i=0;i<sample_count;i++)); do
    random_index=$((RANDOM % (afhq_max_index - afhq_min_index + 1) + afhq_min_index))
    echo "Running qualitative evaluation for sigma = 0.01 for AFHQ dataset"
    # super-res
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/super_res.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
    # deblur
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/deblur.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
    # inpaint pixel
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
    # inpaint box
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
done

for ((i=0;i<sample_count;i++)); do
    random_index=$((RANDOM % (celeba_max_index - celeba_min_index + 1) + celeba_min_index))
    echo "Running qualitative evaluation for sigma = 0.01 for CelebA dataset"
    # super-res
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/super_res.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
    # deblur
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/deblur.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
    # inpaint pixel
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
    # inpaint box
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.0 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
done

# for sigma in ${sigma_list[@]}; do
#     echo "Running qualitative evaluation for sigma = $sigma for AFHQ dataset"
#     # inpaint box
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 109
#     # inpaint pixel
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 53
#     # deblur
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/deblur.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 2
#     # super-res
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/super_res.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 15
# done


# for sigma in ${sigma_list[@]}; do
#     echo "Running qualitative evaluation for sigma = $sigma for CelebA dataset"
#     # inpaint box
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 109
#     # inpaint pixel
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 53
#     # deblur
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/deblur.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 2
#     # super-res
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/super_res.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 15
# done
