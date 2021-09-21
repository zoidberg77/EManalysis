from typing import Tuple, List, Optional
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter

class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size=None, sigma=(1, 1, 1)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample, random_state=np.random.RandomState()):
        blurred = gaussian_filter(sample, sigma=self.sigma)
        return blurred

    def _get_gaussian_kernel2d(self, kernel_size: List[int], sigma: List[float]):
        '''Setup the gaussian kernel in 2d.'''
        kernel1d_x = self._get_gaussian_kernel1d(kernel_size[0], sigma[0])
        kernel1d_y = self._get_gaussian_kernel1d(kernel_size[1], sigma[1])

        return np.matmul(kernel1d_y[:, None], kernel1d_x[None, :])

    def _get_gaussian_kernel1d(self, kernel_size: int, sigma: float):
        '''1d gaussian kernel.'''
        ksize_half = (kernel_size - 1) * 0.5
        x = np.linspace(-ksize_half, ksize_half, kernel_size)
        pdf = np.exp(-0.5 * np.power((x / sigma), 2))

        return pdf / pdf.sum()
