"""Implements various masks for inpainting."""

import os
import math

import argparse
import numpy as np
import logging
import torch


logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description="Create masks for inpainting.")


def square_box_mask(
    center=(0.5, 0.5), width=0.3, height=0.3, D_OR=(3, 256, 256), device="cpu"
):
    """
    Generates a sqaure mask inside the image for inpainting.
    Args:
        - center: Tuple of floats (x, y) representing the center of the square.
        - width: Float representing the width of the square.
        - height: Float representing the height of the square.
        - D_OR: Tuple of integers representing the shape of the image tensor (channels, height, width).
        - device: String representing the device to use; default is 'cpu'.

    Returns:
        - mask: torch.tensor (height, width) representing the mask.
        - missing_indices: torch.tensor (n,) length same as flattened missing part
        - kept_indices: torch.tensor (n,) length same as flattened kept part
        - coordinates_mask: torch.tensor (c*h*w, ), entries are boolean
    """
    channels, h, w = D_OR

    range_width = (
        math.floor((center[0] - width / 2) * D_OR[1]),
        math.ceil((center[0] + width / 2) * D_OR[1]),
    )
    range_height = (
        math.floor((center[1] - height / 2) * D_OR[2]),
        math.ceil((center[1] + width / 2) * D_OR[2]),
    )
    mask = torch.zeros(*D_OR[1:])

    mask[range_width[0] : range_width[1], range_height[0] : range_height[1]] = 1

    missing_r = torch.nonzero(mask.flatten()).long().reshape(-1) * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing_indices = torch.cat([missing_r, missing_g, missing_b], dim=0)

    kept_indices = (
        torch.Tensor([i for i in range(channels * h * w) if i not in missing_indices])
        .to(device)
        .long()
    )

    coordinates_mask = torch.isin(
        torch.arange(math.prod(D_OR), device=device),
        torch.arange(kept_indices.shape[0], device=device),
    )

    return mask, missing_indices, kept_indices, coordinates_mask


def random_pixel_mask(ratio_to_mask=0.3, D_OR=(3, 256, 256), device="cpu"):
    """
    Creates a random pixel-wise mask for inpainting.
    Args:
        - ratio_to_mask: Float representing the ratio of pixels to mask; higher
            values will mask more pixels.
        - D_OR: Tuple of integers representing the shape of the image tensor (channels, height, width).
        - device: String representing the device to use; default is 'cpu'.

    Returns:
        - mask: torch.tensor (height, width) representing the mask.
        - missing_indices: torch.tensor (n,) length same as flattened missing part
        - kept_indices: torch.tensor (n,) length same as flattened kept part
        - coordinates_mask: torch.tensor (c*h*w, ), entries are boolean
    """
    channels, h, w = D_OR

    mask = torch.rand(*D_OR[1:]) < ratio_to_mask

    missing_r = torch.nonzero(mask.flatten()).long().reshape(-1) * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing_indices = torch.cat([missing_r, missing_g, missing_b], dim=0)

    kept_indices = (
        torch.Tensor([i for i in range(channels * h * w) if i not in missing_indices])
        .to(device)
        .long()
    )

    coordinates_mask = torch.isin(
        torch.arange(math.prod(D_OR), device=device),
        torch.arange(kept_indices.shape[0], device=device),
    )

    return mask, missing_indices, kept_indices, coordinates_mask


def load_mask(mask_path, device="cpu"):
    """
    Loads a mask from a .npz file.
    Args:
        - mask_path: String representing the path to the .npz file.
        - device: String representing the device to use; default is 'cpu'.

    Returns:
        - mask: torch.tensor (height, width) representing the mask.
        - missing_indices: torch.tensor (n,) length same as flattened missing part
        - kept_indices: torch.tensor (n,) length same as flattened kept part
        - coordinates_mask: torch.tensor (c*h*w, ), entries are boolean
    """
    loaded_file = np.load(mask_path)
    mask = torch.from_numpy(loaded_file["mask"]).to(device)
    missing_indices = torch.from_numpy(loaded_file["missing_indices"]).long().to(device)
    kept_indices = torch.from_numpy(loaded_file["kept_indices"]).long().to(device)
    coordinates_mask = torch.from_numpy(loaded_file["coordinates_mask"]).to(device)

    return mask, missing_indices, kept_indices, coordinates_mask


def main():
    """
    Creates masks and stores them.
    """
    logging.info("Creating square box mask...")
    mask, missing_indices, kept_indices, coordinates_mask = square_box_mask(
        center=(0.5, 0.5), width=0.3, height=0.3, D_OR=(3, 256, 256), device=DEVICE
    )

    logging.info("Creating random pixel mask...")
    pixel_mask, pixel_missing_indices, pixel_kept_indices, pixel_coordinates_mask = (
        random_pixel_mask(ratio_to_mask=0.9, D_OR=(3, 256, 256), device=DEVICE)
    )

    # save masks to npz files
    os.makedirs("masks", exist_ok=True)
    np.savez(
        "masks/square_box_mask.npz",
        mask=mask.detach().cpu().numpy(),
        missing_indices=missing_indices.detach().cpu().numpy(),
        kept_indices=kept_indices.detach().cpu().numpy(),
        coordinates_mask=coordinates_mask.detach().cpu().numpy(),
    )

    np.savez(
        "masks/random_pixel_mask.npz",
        mask=pixel_mask.detach().cpu().numpy(),
        missing_indices=pixel_missing_indices.detach().cpu().numpy(),
        kept_indices=pixel_kept_indices.detach().cpu().numpy(),
        coordinates_mask=pixel_coordinates_mask.detach().cpu().numpy(),
    )
    logging.info("Masks saved to masks/ directory.")


if __name__ == "__main__":
    main()