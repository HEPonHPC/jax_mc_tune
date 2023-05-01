from typing import List
from .reader import InputReader

import jax.numpy as jnp
import numpy as np

class Filter:
    def __init__(self, name: str):
        self.name = name

    def apply(self, data: InputReader):
        raise NotImplementedError

class Filters:
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def append(self, filter):
        self.filters.append(filter)

    def apply(self, data: InputReader):
        for filter in self.filters:
            data = filter.apply(data)
        return data

class FilterBinsWithLargeValues(Filter):
    def __init__(self, threshold: float = 1000.):
        super().__init__("FilterBinsWithLargeValues")
        self.threshold = threshold

    def apply(self, data: InputReader):
        """Remove bins with values greater than 1000"""
        # <TODO> not a robust way to filter invalid bins

        if data.Y is None:
            raise ValueError("Y is None. Please read the data first.")

        invalid = jnp.array(jnp.any((abs(data.Y[:, :]) > self.threshold), axis=1).nonzero()[0])
        num_of_invalid_bins = len(invalid)

        data.Y = jnp.delete(data.Y, invalid, axis=0)
        data.Y_err = jnp.delete(data.Y_err, invalid, axis=0)
        data.bin_ids = np.delete(data.bin_ids, invalid)
        data.obs_weights = jnp.delete(data.obs_weights, invalid)
        data.num_of_bins = len(data.bin_ids)

        if num_of_invalid_bins > 0:
            print(f"Remove {num_of_invalid_bins} of {self.num_of_orgin_bins} total bins due to their values > 1000.")

        return data
