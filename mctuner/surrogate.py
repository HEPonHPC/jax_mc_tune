from typing import Dict
from .reader import InputReader

import jax
import jax.numpy as jnp

import numpy as np
import scipy.optimize as opt
from .utils_poly_fn import PolyFnUtils

class SurrogateModel:
    def __init__(self,
                 opt_method: str = "BFGS",
                 use_covariance: bool = False,
                 name="SurrogateModel"):
        self.opt_method = opt_method
        self.use_convariance = use_covariance
        self.name = name

        self.hessian_fn = lambda f: jax.jacfwd(jax.jacrev(f))

        # model_weights represents the weights of the surrogate model
        # the key word is reader.name + "_" + bin_idx
        self.model_weights: Dict[str, np.ndarray] = {}
        self.reader = None

        # self.X: np.ndarray = None   # X represents the generator parameters, [num_of_combinations_of_parameters]
        # self.Y: np.ndarray = None   # Y represents the MC simulated values in the bin, [num_of_mc_runs]
        # self.Y_err: np.ndarray = None  # Y_err represents the error of observables, [num_of_mc_runs]

    def set_reader(self, reader):
        self.reader = reader

    def predict(self, model_weight: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def objective(self, model_weight: jnp.ndarray, *args) -> jnp.ndarray:
        raise NotImplementedError

    def gradient(self, model_weight: jnp.ndarray, *args):
        return np.array(jax.grad(self.objective)(model_weight, *args), dtype=np.float64)

    def hessian(self, model_weight: jnp.ndarray, *args):
        return np.array(self.hessian_fn(self.objective)(model_weight, *args), dtype=np.float64)

    def get_bin_str(self, bin_idx: int) -> str:
        if self.reader is None:
            raise ValueError("Reader is not set. Run set_reader() first.")
        return self.reader.name + "_" + str(bin_idx)

    def fit(self, bin_idx: int):
        raise NotImplementedError

    def minimize(self, bin_str: str, *args):
        result = opt.minimize(
            self.objective,
            self.model_weights[bin_str],
            method=self.opt_method,
            args=args,
            jac=self.gradient,
            hess=self.hessian,
            options={'disp': True}
        )
        self.model_weights[bin_str] = result.x
        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            np.save(f, self.model_weights)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model_weights = np.load(f)


class MonomialSurrogateModel(SurrogateModel):
    def __init__(self,
                 order: int = 2,
                 use_mc_error: bool = False,
                 opt_method: str = "BFGS",
                 use_covariance: bool = False,
                 name="monomialSurrogateModel"):
        super().__init__(opt_method, use_covariance, name)
        self.order = order
        self.use_mc_error = use_mc_error

        # VM represents the Vandermonde matrix, [num_of_mc_runs, num_of_combinations_of_parameters]
        # It depends on the number of generator parameters and
        # the order of the monomial function
        # Therefore, the VM matrix is the same for all bins
        self.VM: jnp.ndarray = None

    def set_reader(self, reader):
        super().set_reader(reader)
        poly_fn = PolyFnUtils(self.reader.num_parameters)
        self.VM = poly_fn.vandermonde(self.reader.X, self.order)

    def predict(self, model_weight: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.VM, model_weight)

    def objective(self, model_weight: jnp.ndarray, Y, Y_err) -> jnp.ndarray:
        residule = (self.predict(model_weight) - Y)**2
        result = residule / Y_err**2 if self.use_mc_error and Y_err is not None else residule
        return jnp.sum(result)

    def fit(self, bin_idx: int):
        if self.VM is None:
            raise ValueError("VM is None, please set reader first")

        bin_str = self.get_bin_str(bin_idx)
        self.model_weights[bin_str] = np.zeros(self.VM.shape[1])
        Y = self.reader.Y[bin_idx]
        Y_err = self.reader.Y_err[bin_idx] if self.use_mc_error else None

        self.minimize(bin_str, Y, Y_err)

    def summarize(self):
        print(f"{self.name} Summary:\n"
              f"\tOrder: {self.order}\n"
              f"\tUse MC Error: {self.use_mc_error}\n"
              f"\tOptimization Method: {self.opt_method}\n"
              f"\tUse Covariance: {self.use_convariance}\n"
              f"\tFitted {len(list(self.model_weights.keys()))} Bins\n"
              )
