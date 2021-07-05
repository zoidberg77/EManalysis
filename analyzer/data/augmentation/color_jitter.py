import numpy as np
from typing import Tuple, List, Optional
import torch
from torch import Tensor
import torchvision.transforms.functional as F

class ColorJitter(torch.nn.Module):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self,
                   brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Get the parameters for the randomized transform to be applied on image.
            Args:
                brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                    uniformly. Pass None to turn off the transformation.
                contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                    uniformly. Pass None to turn off the transformation.
                saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                    uniformly. Pass None to turn off the transformation.
                hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                    Pass None to turn off the transformation.

            Returns:
                tuple: The parameters used to apply the randomized transform
                along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def blend(self, vol: np.ndarray, helper: np.ndarray, ratio: float) -> np.ndarray:
        '''applying the specfic blending function to sample vol.'''
        ratio = float(ratio)
        bound = 1.0 if isinstance(vol, np.floating) else 255.0
        return np.clip((ratio * vol + (1.0 - ratio) * helper), 0, bound).astype(vol.dtype)

    def adjust_brightness(self, sample, brightness):
        return self.blend(sample, np.zeros_like(sample), brightness)

    def adjust_contrast(self, sample, contrast):
        dtype = sample.dtype if isinstance(sample, np.floating) else np.float32
        mean = np.mean(sample, keepdims=True)
        return self.blend(sample, mean, contrast)

    def adjust_saturation(self, sample, saturation):
        return self.blend(sample, sample, saturation)

    def adjust_hue(self, sample, hue):
        return sample

    def __call__(self, sample, random_state=np.random.RandomState()):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                sample = self.adjust_brightness(sample, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                sample = self.adjust_contrast(sample, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                sample = self.adjust_saturation(sample, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                sample = self.adjust_hue(sample, hue_factor)

        return sample
