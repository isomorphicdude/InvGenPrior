# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""
import warnings
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from op import upfirdn2d


# migrated from the misc.py in StyleGAN2
# ----------------------------------------------------------------------------
# Symbolic assert.
try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0
# ----------------------------------------------------------------------------
# Context manager to suppress known warnings in torch.jit.trace().


class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return self


# ----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().


def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )


# ----------------------------------------------------------------------------


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    assert_shape(f, [fh, fw][: f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh


# ----------------------------------------------------------------------------


def setup_filter(
    f,
    device=torch.device("cpu"),
    normalize=True,
    flip_filter=False,
    gain=1,
    separable=None,
):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


# Function ported from StyleGAN2
def get_weight(module, shape, weight_var="weight", kernel_init=None):
    """Get/create weight tensor for a convolution or fully-connected layer."""

    return module.param(weight_var, kernel_init, shape)


# buggy implementation for torch
# as torch does not support negative indexing

# class Conv2d(nn.Module):
#   """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

#   def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
#                resample_kernel=(1, 3, 3, 1),
#                use_bias=True,
#                kernel_init=None):
#     super().__init__()
#     assert not (up and down)
#     assert kernel >= 1 and kernel % 2 == 1
#     self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
#     if kernel_init is not None:
#       self.weight.data = kernel_init(self.weight.data.shape)
#     if use_bias:
#       self.bias = nn.Parameter(torch.zeros(out_ch))

#     self.up = up
#     self.down = down
#     self.resample_kernel = resample_kernel
#     self.kernel = kernel
#     self.use_bias = use_bias

#   def forward(self, x):
#     if self.up:
#       x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
#     elif self.down:
#       x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
#     else:
#       x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

#     if self.use_bias:
#       x = x + self.bias.reshape(1, -1, 1, 1)

#     return x


# torch functions
def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


# using CUDA
# def upsample_conv_2d(x, w, k=None, factor=2, gain=1):
#   """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

#      Padding is performed only once at the beginning, not between the
#      operations.
#      The fused op is considerably more efficient than performing the same
#      calculation
#      using standard TensorFlow ops. It supports gradients of arbitrary order.
#      Args:
#        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
#          C]`.
#        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
#          outChannels]`. Grouped convolution can be performed by `inChannels =
#          x.shape[0] // numGroups`.
#        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
#          (separable). The default is `[1] * factor`, which corresponds to
#          nearest-neighbor upsampling.
#        factor:       Integer upsampling factor (default: 2).
#        gain:         Scaling factor for signal magnitude (default: 1.0).

#      Returns:
#        Tensor of the shape `[N, C, H * factor, W * factor]` or
#        `[N, H * factor, W * factor, C]`, and same datatype as `x`.
#   """

#   assert isinstance(factor, int) and factor >= 1

#   # Check weight shape.
#   assert len(w.shape) == 4
#   convH = w.shape[2]
#   convW = w.shape[3]
#   inC = w.shape[1]
#   outC = w.shape[0]

#   assert convW == convH

#   # Setup filter kernel.
#   if k is None:
#     k = [1] * factor
#   k = _setup_kernel(k) * (gain * (factor ** 2))
#   p = (k.shape[0] - factor) - (convW - 1)

#   stride = (factor, factor)

#   # Determine data dimensions.
#   stride = [1, 1, factor, factor]
#   output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
#   output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
#                     output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
#   assert output_padding[0] >= 0 and output_padding[1] >= 0
#   num_groups = _shape(x, 1) // inC

#   # Transpose weights.
#   w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
#   w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
#   # w = w.flip(dims=[3, 4]).permute(0, 2, 1, 3, 4)
#   w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

#   x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)

#   return upfirdn2d(x, torch.tensor(k, device=x.device),
#                    pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


# def _setup_kernel_torch(k):
#     k = np.asarray(k, dtype=np.float32)
#     if k.ndim == 1:
#         k = np.outer(k, k)
#     k /= np.sum(k)
#     return k


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    # seems like this normalization is not implemented in the torch version by NVIDIA
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def upfirdn2d_torch(x, f, up=1, down=1, pad=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops."""
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(pad)

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(
        x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
    )
    x = x[
        :,
        :,
        max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
    ]

    # Setup filter.
    # print(f"before scaling: {f}")
    # print(f"up: {up}")
    # print(f"down: {down}")
    # print(f"gain: {gain}")

    f = f * (gain ** (f.ndim / 2))
    
    # print(f"after scaling: {f}")
    
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    print(f"filter: {f}")
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = torch.nn.functional.conv2d(
            input=x, weight=f.unsqueeze(2), groups=num_channels
        )
        x = torch.nn.functional.conv2d(
            input=x, weight=f.unsqueeze(3), groups=num_channels
        )

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x / torch.sum(f)


# def upsample_2d_torch(x, k=None, factor=2, gain=1):
#     r"""Upsample a batch of 2D images with the given filter.

#     Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
#     and upsamples each image with the given filter. The filter is normalized so
#     that
#     if the input pixels are constant, they will be scaled by the specified
#     `gain`.
#     Pixels outside the image are assumed to be zero, and the filter is padded
#     with
#     zeros so that its shape is a multiple of the upsampling factor.
#     Args:
#         x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
#           C]`.
#         k:            FIR filter of the shape `[firH, firW]` or `[firN]`
#           (separable). The default is `[1] * factor`, which corresponds to
#           nearest-neighbor upsampling.
#         factor:       Integer upsampling factor (default: 2).
#         gain:         Scaling factor for signal magnitude (default: 1.0).

#     Returns:
#         Tensor of the shape `[N, C, H * factor, W * factor]`
#     """
#     assert isinstance(factor, int) and factor >= 1
#     if k is None:
#         k = [1] * factor
#     k = _setup_kernel(k) * (gain * (factor**2))
#     p = k.shape[0] - factor
#     return upfirdn2d_torch(
#         x,
#         torch.tensor(k, device=x.device),
#         up=factor,
#         pad=((p + 1) // 2 + factor - 1, p // 2),
#     )


def upsample_2d_torch(x, f, factor=2, pad=0, flip_filter=False, gain=1):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(factor)
    padx0, padx1, pady0, pady1 = _parse_padding(pad)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d_torch(
        x, f, up=factor, pad=p, flip_filter=flip_filter, gain=gain * upx * upy
    )


def downsample_2d_torch(x, f, factor=2, pad=0, flip_filter=False, gain=1):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the input. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx, downy = _parse_scaling(factor)
    padx0, padx1, pady0, pady1 = _parse_padding(pad)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d_torch(x, f, down=factor, pad=p, flip_filter=flip_filter, gain=gain)


# def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
#   """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

#     Padding is performed only once at the beginning, not between the operations.
#     The fused op is considerably more efficient than performing the same
#     calculation
#     using standard TensorFlow ops. It supports gradients of arbitrary order.
#     Args:
#         x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
#           C]`.
#         w:            Weight tensor of the shape `[filterH, filterW, inChannels,
#           outChannels]`. Grouped convolution can be performed by `inChannels =
#           x.shape[0] // numGroups`.
#         k:            FIR filter of the shape `[firH, firW]` or `[firN]`
#           (separable). The default is `[1] * factor`, which corresponds to
#           average pooling.
#         factor:       Integer downsampling factor (default: 2).
#         gain:         Scaling factor for signal magnitude (default: 1.0).

#     Returns:
#         Tensor of the shape `[N, C, H // factor, W // factor]` or
#         `[N, H // factor, W // factor, C]`, and same datatype as `x`.
#   """

#   assert isinstance(factor, int) and factor >= 1
#   _outC, _inC, convH, convW = w.shape
#   assert convW == convH
#   if k is None:
#     k = [1] * factor
#   k = _setup_kernel(k) * gain
#   p = (k.shape[0] - factor) + (convW - 1)
#   s = [factor, factor]
#   x = upfirdn2d(x, torch.tensor(k, device=x.device),
#                 pad=((p + 1) // 2, p // 2))
#   return F.conv2d(x, w, stride=s, padding=0)


def _shape(x, dim):
    return x.shape[dim]


def upsample_2d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and upsamples each image with the given filter. The filter is normalized so
    that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    # assert isinstance(factor, int) and factor >= 1
    # if k is None:
    #     k = [1] * factor
    # k = _setup_kernel(k) * (gain * (factor**2))
    # p = k.shape[0] - factor
    # return upfirdn2d(
    #     x,
    #     torch.tensor(k, device=x.device),
    #     up=factor,
    #     pad=((p + 1) // 2 + factor - 1, p // 2),
    # )
    return upsample_2d_torch(x, k, factor, gain)


def downsample_2d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 2D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
    and downsamples each image with the given filter. The filter is normalized
    so that
    if the input pixels are constant, they will be scaled by the specified
    `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded
    with
    zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(
        x, torch.tensor(k, device=x.device), down=factor, pad=((p + 1) // 2, p // 2)
    )
