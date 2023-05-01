import h5py

import jax.numpy as jnp
import numpy as np

class InputReader:
    def __init__(self,
                 dtype: jnp.dtype = jnp.float32):
        self.dtype = dtype
        self.X: jnp.ndarray = None  # shape (num_of_mc_runs, num_parameters)
        self.Y: jnp.ndarray = None  # shape (num_of_bins * num_of_mc_runs)
        self.Y_err: jnp.ndarray = None      # shape (num_of_bins * num_of_mc_runs)
        self.param_names: np.ndarray = None  # shape (num_parameters)
        self.bin_ids: np.ndarray = None  # shape (num_of_bins)
        self.obs_weights: np.ndarray = None  # shape (num_of_bins)

        self.num_of_orgin_bins: int = 0
        self.num_parameters: int = 0
        self.num_of_mc_runs: int = 0

    def read_hdf5(self, filename: str):
        f = h5py.File(filename, 'r')

        # generator parameters
        self.X = jnp.array(f["params"][:], dtype=self.dtype)
        self.num_of_mc_runs, self.num_parameters = self.X.shape

        self.param_names = np.array(f['params'].attrs['names']).astype(str)

        # observables
        self.Y = jnp.array(f['values'][:], dtype=self.dtype)
        self.Y_err = jnp.array(f['total_err'][:], dtype=self.dtype)
        self.bin_ids = np.array([x.decode() for x in f.get("index")[:]])
        self.obs_weights = jnp.ones(jnp.shape(self.bin_ids))
        self.num_of_orgin_bins = len(self.bin_ids)

    def read_yoda(self, filename: str):
        raise NotImplementedError

    def read_yoda_from_dir(self, dirname: str):
        raise NotImplementedError

    def __str__(self) -> str:
        return "InputReader: \n" \
               f"  num of measurement bins: {self.num_of_orgin_bins}\n" \
               f"  num of generator paramters: {self.num_parameters}\n" \
               f"  num of MC runs: {self.num_of_mc_runs}\n" \
               f"  dtype: {self.dtype}\n"
