"""
Implements various degredation operators.
Code adapted from RED-diff and MCGdiff.
"""  
from abc import ABC, abstractmethod

import math
import torch
from torchvision import transforms
import einops
import logging
import numpy as np
import ml_collections

logging.basicConfig(level=logging.INFO)

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, config: ml_collections.ConfigDict):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](config)



class H_functions(ABC):
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    
    Here is is assumed the SVD is of the form H = U @ S @ V^T,
    where U and V are orthonormal matrices and S is a rectangular diagonal matrix.
    """

    def __init__(self):
        super(H_functions, self).__init__()

    @abstractmethod
    def V(self, vec):
        """
        Multiplies the input vector by V.
        Returns same shape as input.
        """
        raise NotImplementedError()

    @abstractmethod
    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed. 
        Returns same shape as input.
        """
        raise NotImplementedError()

    @abstractmethod
    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    @abstractmethod
    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    @abstractmethod
    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    @abstractmethod
    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        # print(temp.device)
        singulars = self.singulars()
        # print(singulars.device)
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, : singulars.shape[0]] = temp[:, : singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))
    
    def get_degraded_image(self, vec):
        """
        Returns the degraded image for plotting (B, C, H, W)
        This implementation varies depending on the degradation operator.
        This is a default implementation that works for inpainting and denoising.
        
        Args:  
            - vec (torch.Tensor): The H(x) re-shaped as an image. (B, C*H*W) -> (B, C, H, W)
        """
        # vec = self.H(vec)
        # print(vec.shape)
        # creat a flattend vector to store out for visualisation
        y_0_img = -torch.ones(vec.shape[0], 
                              self.channels * self.img_dim**2, device=self.device)

        # input the observed part
        y_0_img[:, : vec.shape[-1]] = vec

        # re-order back
        y_0_img = self.V(y_0_img)

        # re-shape from flattened vector to image
        y_0_img = y_0_img.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)

        return y_0_img



# Inpainting
@register_operator(name="inpainting")
class Inpainting(H_functions):
    """
    Attributes:  
        - channels: Integer representing the number of channels in the image.
        - img_dim: Integer representing the dimension of the image e.g. 256
        - missing_indices: torch.tensor (n,) representing the missing indices in the image.
        - device: String representing the device to use; default is 'cpu'.
        - kept_indices: torch.tensor (n,) representing the kept indices in the image.
    """
    def __init__(self, config):
        """
        Initialize from config.degredation.
        """
        super(Inpainting, self).__init__()
        
        channels = config.channels
        img_dim = config.img_dim
        missing_indices = config.missing_indices
        device = config.device
        self.device = device
        
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.nn.Parameter(
            torch.ones(channels * img_dim**2 - missing_indices.shape[0]).to(device),
            requires_grad=False,
        )
        self.missing_indices = torch.nn.Parameter(missing_indices, requires_grad=False)
        self.kept_indices = torch.nn.Parameter(
            torch.Tensor(
                [i for i in range(channels * img_dim**2) if i not in missing_indices]
            )
            .to(device)
            .long(),
            requires_grad=False,
        )

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1).to(self.device)
        out = torch.zeros_like(temp).to(self.device)
        out[:, self.kept_indices] = temp[:, : self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0] :]
        return (
            out.reshape(vec.shape[0], -1, self.channels)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

    def Vt(self, vec):
        temp = (
            vec.clone()
            .reshape(vec.shape[0], self.channels, -1)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        ).to(self.device)
        out = torch.zeros_like(temp).to(self.device)
        out[:, : self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0] :] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=vec.device
        )
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp
    
    
    # def get_degraded_image(self, vec):
    #     # creat a flattend vector to store out for visualisation
    #     y_0_img = -torch.ones(self.channels * self.img_dim**2, device=self.device)

    #     # input the observed part
    #     y_0_img[: vec.shape[-1]] = vec[0]

    #     # re-order back
    #     y_0_img = self.V(y_0_img[None, ...])

    #     # re-shape from flattened vector to image
    #     y_0_img = y_0_img.reshape(self.channels, self.img_dim, self.img_dim)

    #     return y_0_img
    
    # override the H function to make it more efficient for batched operations
    # and matrices so (B, d_x, d_x) -> (B, d_y, d_x)
    # def H(self, mat):
    #     """
    #     Applies the H function to each column of the matrix.
    #     which is of shape (B, d_x, d_x). 
        
    #     Determine the input shape mat or vec.
        
    #     Returns a matrix of shape (B, d_x, d_y)
        
    #     Note: 
    #         - We assume d_x = channel * height * width so that
    #         the input matrix columns are flattened images.  
    #         - The operation is simply extract the kept indices entries
    #         for each column and reshape back
    #     """
    #     if len(mat.shape) == 2:
    #         assert mat.shape[1] == self.channels * self.img_dim**2
    #         return mat[:, self.kept_indices]
    #     elif len(mat.shape) == 3:
    #         assert mat.shape[1] == mat.shape[2] == self.channels * self.img_dim**2
    #         return mat[:, :, self.kept_indices]
    #     else:
    #         raise ValueError("Input shape not recognized.")
        
        
    # def right_multiply_Ht(self, mat):
    #     """
    #     Computes mat @ H^t or vec @ H^t.
    #     where H is (d_y, d_x) and H^t is (d_x, d_y)
    #     and mat is (B, A, d_y) or vec is (B, d_y) respectively.  
        
    #     Note:
    #         - This is implemented as first left multiply and then transpose
    #         mat @ H^t = (H @ mat^t)^t
    #     """
    #     if len(mat.shape) == 2:
    #         assert mat.shape[1] == self.channels * self.img_dim**2
    #         # return mat[:, self.kept_indices].transpose(1, 0)
    #         raise NotImplementedError("Not implemented for vec yet.")
    #     elif len(mat.shape) == 3:
    #         # assert mat.shape[1] == mat.shape[2] == self.channels * self.img_dim**2
    #         return mat[:, self.kept_indices, :].transpose(1, 2)
    #     else:
    #         raise ValueError("Input shape not recognized.")
        
        
    # def quadratic_form(self, mat):
    #     """
    #     Computes H @ mat @ H^t. No vector operation here.
    #     where H is (d_y, d_x) and H^t is (d_x, d_y)
    #     so mat is (B, d_x, d_x) and the output is (B, d_y, d_y)  
        
    #     Note:
    #         - This is implemented as the previous two operations combined.
    #     """
    #     assert len(mat.shape) >= 3 
    #     assert mat.shape[1] == mat.shape[2] == self.channels * self.img_dim**2
        
    #     return self.right_multiply_Ht(self.H(mat))
    
    # # override again
    # def Ht(self, mat):
    #     """
    #     Applies the H^t function to each column of the matrix.
    #     mat has shape (B, d_y, d_y)
    #     and output has shape (B, d_x, d_y)
    #     """
    #     # fill the missing entries with zeros
        
    #     if len(mat.shape) == 2:
    #         assert mat.shape[1] == len(self.kept_indices)
    #         temp = torch.zeros(
    #             (mat.shape[0], self.channels * self.img_dim**2), 
    #             device=mat.device
    #         )
    #         temp[:, self.kept_indices] = mat
    #         return temp
        
    #     elif len(mat.shape) == 3:
    #         assert mat.shape[2] == len(self.kept_indices)
    #         temp = torch.zeros(
    #             (mat.shape[0], mat.shape[1], self.channels * self.img_dim**2), 
    #             device=mat.device
    #         )
    #         temp[:, :, self.kept_indices] = mat
    #         return temp
    
    
# Denoising
@register_operator(name="denoising")
class Denoising(H_functions):
    def __init__(self, config):
        
        channels = config.channels
        img_dim = config.img_dim
        device = config.device
        self.img_dim = img_dim
        self.channels = channels
        self.device = device
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec, for_H=True):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


# Super Resolution frmo NVIDIA
@register_operator(name="super_resolution")
class SuperResolution(H_functions):
    """
    Degrades the image to a lower resolution.
    (B, C, H, W) -> (B, C, H//ratio, W//ratio)
    This implementation is simply taking the average of pixels in a block of size ratio x ratio.
    """
    def __init__(self, config):  # ratio = 2 or 4
        super(SuperResolution, self).__init__()
        
        img_dim = config.img_dim
        ratio = config.ratio
        channels = config.channels
        device = config.device
        
        assert img_dim % ratio == 0
        
        self.img_dim = img_dim
        self.channels = channels
        self.device = device
        
        # dimension of the small image (after downsampling)
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.U_small = self.U_small.to(device)
        self.V_small = self.V_small.to(device)
        self.singulars_small = self.singulars_small.to(device)
        self.Vt_small = self.V_small.transpose(0, 1).to(device)
        

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim
        )
        
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        ) # (B, C, y_dim, y_dim, ratio, ratio)
        # after first unfold is (B, C, y_dim, img_dim, ratio)
        # after second unfold is (B, C, y_dim, y_dim, ratio, ratio)
        
        # patches = patches.contiguous().reshape(
        #     vec.shape[0], self.channels, -1, self.ratio**2
        # )
        
        patches = patches.reshape(
             vec.shape[0], self.channels, -1, self.ratio**2
        ).to(self.device)
        
        # multiply each by the small V transposed
        # after reshaped patches are (B*C*y_dim**2, ratio**2, 1)
        patches = torch.matmul(
            self.Vt_small.to(self.device), patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        
        # after multiplication (B*C*y_dim**2, ratio**2, 1)
        # after reshape (B, C, y_dim**2, ratio**2)
        
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(
            vec.shape[0], self.channels * self.img_dim**2, device=self.device
        )
        recon[:, : self.channels * self.y_dim**2] = patches[:, :, :, 0].view(
            vec.shape[0], self.channels * self.y_dim**2
        )
        for idx in range(self.ratio**2 - 1):
            recon[:, (self.channels * self.y_dim**2 + idx) :: self.ratio**2 - 1] = (
                patches[:, :, :, idx + 1].view(
                    vec.shape[0], self.channels * self.y_dim**2
                )
            )
        return recon.to(self.device)
    
    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        
        # create zeros to fill (B, C, y_dim**2, ratio**2)
        patches = torch.zeros(
            vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device
        )
        # fill each 
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_dim**2].view(
            vec.shape[0], self.channels, -1
        )
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[
                :, (self.channels * self.y_dim**2 + idx) :: self.ratio**2 - 1
            ].view(vec.shape[0], self.channels, -1)
            
        # multiply each patch by the small V
        patches = torch.matmul(
            self.V_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # repatch the patches into an image
        patches_orig = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        # assumes vec is lower dim
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp
    
    def get_degraded_image(self, vec):
        # assumes vec is full resolution
       """
       Returns the lower-resolution image for plotting (B, C, H//ratio, W//ratio)
       """
       
       return vec.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim)
    
    



# Colorization
@register_operator(name="colorization")
class Colorization(H_functions):
    """
    Degrades the RGB image to grayscale.  
    Shape (B, C, H, W) -> (B, 1, H, W)
    """
    def __init__(self, config):
        super(Colorization, self).__init__()
        
        channels = config.channels
        img_dim = config.img_dim
        device = config.device
        
        assert channels == 3
        self.channels = channels
        self.img_dim = img_dim
        self.device = device
        
        # Do the SVD for the per-pixel matrix
        # H = torch.nn.Parameter(
        #     torch.Tensor([[0.3333, 0.3333, 0.3333]]), requires_grad=False
        # ).to(device)
        H = torch.nn.Parameter(
            torch.Tensor([[0.299, 0.587, 0.114]]), requires_grad=False
        ).to(device)
        
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1).to(device)
        # self.Vt_small = self.Vt_small.to(device)
        self.V_small = self.V_small.to(device)
        self.U_small = self.U_small.to(device)
        self.singulars_small = self.singulars_small.to(device)
        

    def V(self, vec):
        # get the needles [R, G, B]
        needles = (
            vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).to(self.device)
        )  # shape: B, WH, C'
        
        # multiply each needle by the small V
        needles = torch.matmul(
            self.V_small.to(self.device), needles.reshape(-1, self.channels, 1)
        ).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C
        
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).to(self.device)
        )  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(
            self.Vt_small.to(self.device), needles.reshape(-1, self.channels, 1)
        ).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=self.device
        )
        temp[:, : self.img_dim**2] = reshaped
        return temp
    
    def get_degraded_image(self, vec):
        return vec.reshape(vec.shape[0], 1, self.img_dim, self.img_dim)



# Deblurring
@register_operator(name="deblurring2")
class Deblurring2(H_functions):
    def __init__(self, config):
        # a custom implementation using torchvision so can match with Pokle et al. 2024
        kernel_size = config.kernel_size
        channels = config.channels
        img_dim = config.img_dim
        device = config.device
        intensity = config.intensity # intensity of the blur, sigma of the gaussian
        
        self.img_dim = img_dim
        self.channels = channels
        self.device = device
        # do not use anisotropic blurring for now
        # this is implemented by Cardoso et al. 2023 in Deblurring2D
        self.gaussian_blur_torch = transforms.GaussianBlur(kernel_size, sigma=intensity).to(device)
        
        
    def H(self, vec):
        return self.gaussian_blur_torch(vec).to(self.device)
    
    def get_degraded_image(self, vec):
        return vec
        
    def V(self, vec):
        raise NotImplementedError("Not implemented for deblurring.")
    
    def Vt(self, vec):
        raise NotImplementedError("Not implemented for deblurring.")
    
    def U(self, vec):
        raise NotImplementedError("Not implemented for deblurring.")
    
    def Ut(self, vec):
        raise NotImplementedError("Not implemented for deblurring.")
    
    def singulars(self):
        raise NotImplementedError("Not implemented for deblurring.")
    
    def add_zeros(self, vec):
        raise NotImplementedError("Not implemented for deblurring.")
    
    


@register_operator(name="gmm_h")
class H_func_gmm(H_functions):
    def __init__(self, config):
        obs_dim = config.obs_dim
        dim = config.dim
        H_mat = config.H_mat
        V_mat = config.V_mat
        U_mat = config.U_mat
        singulars = config.singulars
        
        if H_mat is None:
            H_mat = torch.randn(obs_dim * dim).reshape(obs_dim, dim)
            U, S, Vt = torch.linalg.svd(H_mat, full_matrices=True)
            coordinate_mask = torch.ones_like(Vt[0])
            coordinate_mask[len(S):] = 0
            
            # sampling Unif[0, 1] for the singular values
            diag = torch.sort(torch.rand_like(S), descending=True).values
            
            H_mat = U @ (torch.diag(diag)) @ Vt[coordinate_mask==1, :]
        else:
            H_mat = H_mat
        
        self.dim = dim
        self.obs_dim = obs_dim
        self.H_mat = H_mat
        self.V_mat = V_mat
        self.U_mat = U_mat
        self._singulars = singulars
        
    def H(self, vec):
        return torch.einsum("ij,bj->bi", self.H_mat, vec)
    
    def get_degraded_image(self, vec):
        return vec
        
    def V(self, vec):
        return torch.einsum("ij,bj->bi", self.V_mat, vec)
    
    def Vt(self, vec):
        return torch.einsum("ij,bj->bi", self.V_mat.T, vec)
    
    def U(self, vec):
        return torch.einsum("ij,bj->bi", self.U_mat, vec)
    
    def Ut(self, vec):
        return torch.einsum("ij,bj->bi", self.U_mat.T, vec)
    
    def singulars(self):
        return self._singulars
    
    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.dim))
        temp[:, :vec.shape[1]] = vec
        return temp        

@register_operator(name="deblurring")
class Deblurring(H_functions):
    # TODO: can use einops to make the code more readable
    def mat_by_img(self, M, v):
        """
        Returns the matrix-vector product M @ v.
        M is of shape (d_y, d_x) and v is of shape (B, d_x).
        """
        return torch.matmul(
            M, v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim)
        ).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, N):
        """
        Returns the vector-matrix product v @ M.
        v is of shape (B, d_x) and M is of shape (d_x, d_y).
        """
        return torch.matmul(
            v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim), N
        ).reshape(v.shape[0], self.channels, self.img_dim, N.shape[1])

    def __init__(self, config):
        kernel = config.kernel
        channels = config.channels
        img_dim = config.img_dim
        device = config.device
        self.device = device
        try:
            ZERO = config.ZERO
        except:
            ZERO = 3e-2
        
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue # outside the image
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        
        # move to device
        self.U_small = self.U_small.to(device)
        self.singulars_small = self.singulars_small.to(device)
        self.V_small = self.V_small.to(device)
        
        #ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        vec = vec.to(self.device) # added
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out.to(self.device)

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        vec = vec.to(self.device) # added
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1).to(self.device)

    def U(self, vec):
        vec = vec.to(self.device)
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out.to(self.device)

    def Ut(self, vec):
        vec = vec.to(self.device)
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1).to(self.device)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1).to(self.device)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1).to(self.device)
    
    def get_degraded_image(self, vec):
        # print(vec)
        return vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)



# Anisotropic Deblurring
class Deblurring2D(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(
            M, v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim)
        ).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(
            v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim), M
        ).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        super(Deblurring2D, self).__init__()
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0] // 2, i + kernel1.shape[0] // 2):
                if j < 0 or j >= img_dim:
                    continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0] // 2]
        # build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0] // 2, i + kernel2.shape[0] // 2):
                if j < 0 or j >= img_dim:
                    continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(
            H_small1, some=False
        )
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(
            H_small2, some=False
        )
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0

        self.U_small1, self.U_small2 = torch.nn.Parameter(
            self.U_small1, requires_grad=False
        ), torch.nn.Parameter(self.U_small2, requires_grad=False)
        self.singulars_small1 = torch.nn.Parameter(
            self.singulars_small1, requires_grad=False
        )
        self.singulars_small2 = torch.nn.Parameter(
            self.singulars_small2, requires_grad=False
        )
        self.V_small1 = torch.nn.Parameter(self.V_small1, requires_grad=False)
        self.V_small2 = torch.nn.Parameter(self.V_small2, requires_grad=False)

        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small1.reshape(img_dim, 1),
            self.singulars_small2.reshape(1, img_dim),
        ).reshape(img_dim**2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(
            descending=True
        )  # , stable=True)
        self._singulars = torch.nn.Parameter(self._singulars, requires_grad=False)
        self._perm = torch.nn.Parameter(self._perm, requires_grad=False)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(
            vec.shape[0], -1
        )
        return out

    def Vt(self, vec, for_H=True):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(
            vec.shape[0], -1
        )
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, self.channels).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

# Convolution-based super-resolution
class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim, dim)).reshape(
            v.shape[0], self.channels, M.shape[0], dim
        )

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim, dim), M).reshape(
            v.shape[0], self.channels, dim, M.shape[1]
        )

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        # build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                # reflective padding
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= img_dim:
                    j_effective = (img_dim - 1) - (j_effective - img_dim)
                # matrix building
                H_small[i // stride, j_effective] += kernel[
                    j - i + kernel.shape[0] // 2
                ]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(
            H_small, some=False
        )
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small.reshape(small_dim, 1),
            self.singulars_small.reshape(1, small_dim),
        ).reshape(small_dim**2)
        # permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = (
            torch.Tensor(
                [
                    self.img_dim * i + j
                    for i in range(self.small_dim)
                    for j in range(self.small_dim)
                ]
                + [
                    self.img_dim * i + j
                    for i in range(self.small_dim)
                    for j in range(self.small_dim, self.img_dim)
                ]
            )
            .to(device)
            .long()
        )

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )[:, : self._perm.shape[0], :]
        temp[:, self._perm.shape[0] :, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )[:, self._perm.shape[0] :, :]
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(
            vec.shape[0], -1
        )
        return out

    def Vt(self, vec, for_H=True):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries
        temp[:, :, : self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.small_dim**2, self.channels, device=vec.device
        )
        temp[:, : self.small_dim**2, :] = vec.clone().reshape(
            vec.shape[0], self.small_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(
            out, self.U_small.transpose(0, 1), self.small_dim
        ).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(
            self.U_small.transpose(0, 1), vec.clone(), self.small_dim
        )
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp
