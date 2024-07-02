import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from physics.operators import Inpainting
from physics.create_mask import random_pixel_mask, square_box_mask


# inpainting tests
mask, missing_indices, kept_indices, coordinates_mask = random_pixel_mask(
    ratio_to_mask=0.1, D_OR=(3, 8, 8), device="cpu"
)

mask, missing_indices, kept_indices, coordinates_mask = square_box_mask(
    center=(0.5, 0.5), width=0.3, height=0.3, D_OR=(3, 8, 8), device="cpu"
)


inpainting_pixel = Inpainting(
    channels=3, img_dim=8, missing_indices=missing_indices, device="cpu"
)

inpainting_box = Inpainting(
    channels=3, img_dim=8, missing_indices=missing_indices, device="cpu"
)


def test_inpainting_functions_vec():
    test_img = torch.randn(2, 3, 8, 8)
    flattened_img = test_img.reshape(test_img.shape[0], -1)
    assert inpainting_pixel.H(flattened_img).shape == (
        2,
        len(inpainting_pixel.kept_indices),
    )
    assert inpainting_box.H(flattened_img).shape == (
        2,
        len(inpainting_box.kept_indices),
    )


def test_inpainting_functions_mat():
    test_mat = torch.randn(2, 3 * 8 * 8, 3 * 8 * 8)
    assert (
        inpainting_pixel.Ht(inpainting_pixel.right_multiply_Ht(test_mat)).shape
        == test_mat.shape
    )
    assert (
        inpainting_box.Ht(inpainting_box.right_multiply_Ht(test_mat)).shape
        == test_mat.shape
    )
    assert inpainting_pixel.quadratic_form(test_mat).shape == (
        2,
        len(inpainting_pixel.kept_indices),
        len(inpainting_pixel.kept_indices),
    )
    assert inpainting_box.quadratic_form(test_mat).shape == (
        2,
        len(inpainting_box.kept_indices),
        len(inpainting_box.kept_indices),
    )
