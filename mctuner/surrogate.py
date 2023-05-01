
from .reader import InputReader

import jax
import jax.numpy as jnp

import numpy as np
import scipy.optimize as opt

class SurrogateModel:
    def __init__(self,
                 opt_method: str = "BFGS",
                 name="SurrogateModel"):
        self.name = name
        self.hessian_fn = lambda f: jax.jacfwd(jax.jacrev(f))
        self.model_weights: np.ndarray = None

    def predict(self, reader: InputReader) -> jnp.ndarray:
        raise NotImplementedError

    def objective(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def gradient(self, X: jnp.ndarray):
        return np.array(jax.grad(self.objective)(X), dtype=np.float64)

    def hessian(self, X: jnp.ndarray):
        return np.array(self.hessian_fn(self.objective)(X), dtype=np.float64)

    def fit(self, reader: InputReader):
        raise NotImplementedError

    def minimize(self):
        result = opt.minimize(
            self.objective,
            self.model_weights,
            method=self.opt_method,
            jac=self.gradient,
            hess=self.hessian,
            options={'disp': True}
        )
        self.model_weights = result.x
        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            np.save(f, self.model_weights)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model_weights = np.load(f)
