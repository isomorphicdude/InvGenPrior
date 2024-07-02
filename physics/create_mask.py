"""Implements various masks for inpainting."""

import math
import torch


def square_box_mask(
    center=(0.5, 0.5), width=0.3, height=0.3, D_OR=(3, 256, 256), device='cpu'
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


def random_pixel_mask(ratio_to_mask = 0.3,
                      D_OR = (3, 256, 256),
                      device='cpu'):
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

    kept_indices = torch.Tensor([i for i in range(channels * h * w) if i not in missing_indices]).to(device).long()

    coordinates_mask = torch.isin(
        torch.arange(math.prod(D_OR), device=device),
        torch.arange(kept_indices.shape[0], device=device))


    return mask, missing_indices, kept_indices, coordinates_mask