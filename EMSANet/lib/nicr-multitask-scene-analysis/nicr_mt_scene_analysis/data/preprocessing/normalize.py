# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np

from ...types import BatchType
 

def normalize(
    value: np.array,
    mean: np.array,
    std: np.array,
    dtype: str = 'float32',
    inplace: bool = False
) -> np.array:
    if value.dtype != dtype:
        # convert dtype (if does not match, we have to copy)
        value = value.astype(dtype, copy=True)
    else:
        if not inplace:
            value = value.copy()

    # apply normalization inplace (add spatial axes before)
    value -= mean[np.newaxis, np.newaxis, ...]
    value /= std[np.newaxis, np.newaxis, ...]

    return value


class NormalizeRGB:
    def __init__(
        self,
        output_dtype: str = 'float32'
    ) -> None:
        self._output_dtype = output_dtype

        # prepare normalization parameters
        # RGB: values taken from torchvision (= ImageNet mean and std)
        self._rgb_mean = np.array((0.485, 0.456, 0.406),
                                  dtype=self._output_dtype) * 255
        self._rgb_std = np.array((0.229, 0.224, 0.225),
                                 dtype=self._output_dtype) * 255

    def __call__(self, sample: BatchType) -> BatchType:
        assert sample['rgb'].dtype == 'uint8'
        sample['rgb'] = normalize(sample['rgb'],
                                  mean=self._rgb_mean,
                                  std=self._rgb_std,
                                  dtype=self._output_dtype,
                                  inplace=False)

        return sample

class NormalizeDepth:
    def __init__(
        self,
        depth_mean: float,
        depth_std: float,
        raw_depth: bool = False,
        output_dtype: str = 'float32'
    ) -> None:
        assert depth_std != 0.0

        self._raw_depth = raw_depth
        self._output_dtype = output_dtype

        # prepare normalization parameters
        self._depth_mean = np.array(depth_mean, dtype=self._output_dtype)
        self._depth_std = np.array(depth_std, dtype=self._output_dtype)

    def __call__(self, sample: BatchType) -> BatchType:
        if self._raw_depth:
            # get mask of invalid depth values
            invalid_mask = sample['depth'] == 0

        sample['depth'] = normalize(sample['depth'],
                                    mean=self._depth_mean,
                                    std=self._depth_std,
                                    dtype=self._output_dtype,
                                    inplace=False)

        if self._raw_depth:
            # reset invalid values (the network should not be able to learn
            # from invalid values)
            sample['depth'][invalid_mask] = 0

        return sample


    


class NormalizeDepth_anything:
    def __init__(
        self,
        depth_mean: float,
        depth_std: float,
        raw_depth: bool = False,
        output_dtype: str = 'float32'
    ) -> None:
        assert depth_std != 0.0

        self._raw_depth = raw_depth
        self._output_dtype = output_dtype
        depth_mean =  78.88898035000655
        depth_std = 66.41249573741005
        # prepare normalization parameters
        self._depth_mean = np.array(depth_mean, dtype=self._output_dtype)
        self._depth_std = np.array(depth_std, dtype=self._output_dtype)

    def __call__(self, sample: BatchType) -> BatchType:
        # import pdb;pdb.set_trace()

        sample['depth_anything'] = normalize(sample['depth_anything'],
                                    mean=self._depth_mean,
                                    std=self._depth_std,
                                    dtype=self._output_dtype,
                                    inplace=False)
        # sample['depth_anything'] = sample['depth_anything'].astype(np.float32) / 255.0

        return sample

class NormalizeDepth_meta:
    def __init__(
        self,
        depth_mean: float,
        depth_std: float,
        raw_depth: bool = False,
        output_dtype: str = 'float32'
    ) -> None:
        assert depth_std != 0.0

        self._raw_depth = raw_depth
        self._output_dtype = output_dtype
        depth_mean =  2643.0247035139873
        depth_std = 1540.6424288762523
        # prepare normalization parameters
        self._depth_mean = np.array(depth_mean, dtype=self._output_dtype)
        self._depth_std = np.array(depth_std, dtype=self._output_dtype)

    def __call__(self, sample: BatchType) -> BatchType:

        sample['depth_meta'] = normalize(sample['depth_meta'],
                                    mean=self._depth_mean,
                                    std=self._depth_std,
                                    dtype=self._output_dtype,
                                    inplace=False)


        return sample
class NormalizeDepth_mari:
    def __init__(
        self,
        depth_mean: float,
        depth_std: float,
        raw_depth: bool = False,
        output_dtype: str = 'float32'
    ) -> None:
        assert depth_std != 0.0

        self._raw_depth = raw_depth
        self._output_dtype = output_dtype
        depth_mean =  29620.396526414275
        depth_std = 18208.276937066068
        # prepare normalization parameters
        self._depth_mean = np.array(depth_mean, dtype=self._output_dtype)
        self._depth_std = np.array(depth_std, dtype=self._output_dtype)

    def __call__(self, sample: BatchType) -> BatchType:

        sample['depth_mari'] = normalize(sample['depth_mari'],
                                    mean=self._depth_mean,
                                    std=self._depth_std,
                                    dtype=self._output_dtype,
                                    inplace=False)


        return sample
