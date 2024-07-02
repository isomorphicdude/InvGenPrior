"""
Implements various degredation operators.
Code adapted from RED-diff and MCGdiff, which is adapted from RED-diff.
"""  

import torch
import einops


class H_functions(torch.nn.Module):
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def __init__(self):
        super(H_functions, self).__init__()

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec, for_H=True):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

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
        singulars = self.singulars()
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



# Inpainting
class Inpainting(H_functions):
    """
    Attributes:  
        - channels: Integer representing the number of channels in the image.
        - img_dim: Integer representing the dimension of the image e.g. 256
        - missing_indices: torch.tensor (n,) representing the missing indices in the image.
        - device: String representing the device to use; default is 'cpu'.
        - kept_indices: torch.tensor (n,) representing the kept indices in the image.
    """
    def __init__(self, channels, img_dim, missing_indices, device):
        super(Inpainting, self).__init__()
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
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
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
        )
        out = torch.zeros_like(temp)
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
    
    # override the H function to make it more efficient for batched operations
    # and matrices so (B, d_x, d_x) -> (B, d_y, d_x)
    def H(self, mat):
        """
        Applies the H function to each column of the matrix.
        which is of shape (B, d_x, d_x). 
        
        Determine the input shape mat or vec.
        
        Returns a matrix of shape (B, d_x, d_y)
        
        Note: 
            - We assume d_x = channel * height * width so that
            the input matrix columns are flattened images.  
            - The operation is simply extract the kept indices entries
            for each column and reshape back
        """
        if len(mat.shape) == 2:
            assert mat.shape[1] == self.channels * self.img_dim**2
            return mat[:, self.kept_indices]
        elif len(mat.shape) == 3:
            assert mat.shape[1] == mat.shape[2] == self.channels * self.img_dim**2
            return mat[:, :, self.kept_indices]
        else:
            raise ValueError("Input shape not recognized.")
        
        
    def right_multiply_Ht(self, mat):
        """
        Computes mat @ H^t or vec @ H^t.
        where H is (d_y, d_x) and H^t is (d_x, d_y)
        and mat is (B, A, d_y) or vec is (B, d_y) respectively.  
        
        Note:
            - This is implemented as first left multiply and then transpose
            mat @ H^t = (H @ mat^t)^t
        """
        if len(mat.shape) == 2:
            assert mat.shape[1] == self.channels * self.img_dim**2
            # return mat[:, self.kept_indices].transpose(1, 0)
            raise NotImplementedError("Not implemented for vec yet.")
        elif len(mat.shape) == 3:
            # assert mat.shape[1] == mat.shape[2] == self.channels * self.img_dim**2
            return mat[:, self.kept_indices, :].transpose(1, 2)
        else:
            raise ValueError("Input shape not recognized.")
        
        
    def quadratic_form(self, mat):
        """
        Computes H @ mat @ H^t. No vector operation here.
        where H is (d_y, d_x) and H^t is (d_x, d_y)
        so mat is (B, d_x, d_x) and the output is (B, d_y, d_y)  
        
        Note:
            - This is implemented as the previous two operations combined.
        """
        assert len(mat.shape) >= 3 
        assert mat.shape[1] == mat.shape[2] == self.channels * self.img_dim**2
        
        return self.right_multiply_Ht(self.H(mat))
    
    # override again
    def Ht(self, mat):
        """
        Applies the H^t function to each column of the matrix.
        mat has shape (B, d_y, d_y)
        and output has shape (B, d_x, d_y)
        """
        # fill the missing entries with zeros
        
        if len(mat.shape) == 2:
            assert mat.shape[1] == len(self.kept_indices)
            temp = torch.zeros(
                (mat.shape[0], self.channels * self.img_dim**2), 
                device=mat.device
            )
            temp[:, self.kept_indices] = mat
            return temp
        
        elif len(mat.shape) == 3:
            assert mat.shape[2] == len(self.kept_indices)
            temp = torch.zeros(
                (mat.shape[0], mat.shape[1], self.channels * self.img_dim**2), 
                device=mat.device
            )
            temp[:, :, self.kept_indices] = mat
            return temp