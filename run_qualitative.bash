#!/bin/bash

# Run the qualitative evaluation
# sigma 0.0, 0.01, 0.05
# sample_N 50, 100
# inpaint box 52
# inpaint pixel 53
# deblur 2
# super-res 15

# sigma_list=(0.0 0.01 0.05)
sigma_list=(0.01 1.0)
sample_N=100
sample_count=10

afhq_min_index=0
afhq_max_index=499

celeba_min_index=0
celeba_max_index=1999

# for ((i=0;i<sample_count;i++)); do
#     random_index=$((RANDOM % (afhq_max_index - afhq_min_index + 1) + afhq_min_index))
#     echo "Running qualitative evaluation for sigma = 0.01 for AFHQ dataset"
#     echo "Random index: $random_index"
#     # super-res
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/super_res.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
#     # deblur
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/deblur.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
#     # inpaint pixel
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
#     # inpaint box
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
# done

# for ((i=0;i<sample_count;i++)); do
#     random_index=$((RANDOM % (celeba_max_index - celeba_min_index + 1) + celeba_min_index))
#     echo "Running qualitative evaluation for sigma = 0.01 for CelebA dataset"
#     echo "Random index: $random_index"
#     # super-res
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/super_res.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
#     # deblur
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/deblur.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
#     # inpaint pixel
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
#     # inpaint box
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N 100 --data_index "$random_index" --sampling_var 0.01 --clamp_to 1.0 --noise_sigma 0.01 --max_iter 5
# done

# for sigma in ${sigma_list[@]}; do
#     echo "Running qualitative evaluation for sigma = $sigma for AFHQ dataset"
#     # inpaint box
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 109 --clamp_to 1.0 --max_iter 1
#     # inpaint pixel
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 53 --clamp_to 1.0 --max_iter 1
#     # deblur
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/deblur.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 2 --clamp_to 1.0 --max_iter 1
#     # super-res
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/super_res.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 15 --clamp_to 1.0 --max_iter 1
# done


# for sigma in ${sigma_list[@]}; do
#     echo "Running qualitative evaluation for sigma = $sigma for CelebA dataset"
#     # inpaint box
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 109 --clamp_to 1.0 --max_iter 1 --sampling_var 0.0 
#     # inpaint pixel
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 1253 --clamp_to 1.0 --max_iter 1 --sampling_var 0.0
#     # deblur
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/deblur.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 1547 --clamp_to 1.0 --max_iter 1 --sampling_var 0.0
#     # super-res
#     python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/super_res.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma $sigma --data_index 389 --clamp_to 1.0 --max_iter 1 --sampling_var 0.0
# done

# run for max_iter 3, 5, 10
max_iter_list=(3 5 10)
for max_iter in ${max_iter_list[@]}; do
    echo "Running qualitative evaluation for max_iter = $max_iter for AFHQ dataset"
    # inpaint box
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 109 --clamp_to 1.0 --max_iter $max_iter --compare_iter
    # inpaint pixel
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 53 --clamp_to 1.0 --max_iter $max_iter --compare_iter
    # deblur
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/deblur.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 15 --clamp_to 1.0 --max_iter $max_iter --compare_iter
    # super-res
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/afhq/super_res.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 2 --clamp_to 1.0 --max_iter $max_iter --compare_iter
done

for max_iter in ${max_iter_list[@]}; do
    echo "Running qualitative evaluation for max_iter = $max_iter for CelebA dataset"
    # inpaint box
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_box.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 109 --clamp_to 1.0 --max_iter $max_iter --sampling_var 0.0  --compare_iter
    # inpaint pixel
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/inpaint_pixel.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 1253 --clamp_to 1.0 --max_iter $max_iter --sampling_var 0.0 --compare_iter
    # deblur
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/deblur.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 1547 --clamp_to 1.0 --max_iter $max_iter --sampling_var 0.0 --compare_iter
    # super-res
    python run_qualitative.py --config /content/InvGenPrior/configs/tmpd/celeba/super_res.py --workdir /content/InvGenPrior/ --sample_N $sample_N --noise_sigma 0.01 --data_index 389 --clamp_to 1.0 --max_iter $max_iter --sampling_var 0.0 --compare_iter
done
