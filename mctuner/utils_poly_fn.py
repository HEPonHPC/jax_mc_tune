from typing import Union
import numpy as np
import jax.numpy as jnp

from . import monomial


class PolyFnUtils:
    def __init__(self,
                 num_params: int,
                 name="PolyFnUtils"):
        self.num_params = num_params
        self.name = name


    def vandermonde(self,
                    array: Union[np.ndarray, jnp.ndarray],
                    order: int) -> np.ndarray:
        """Construct the Vandermonde matrix.
        array: 2D array of shape (num_parameters, num_of_mc_runs)

        order: int
            the maximum order of the monomial.

        Returns
        -------
        V: 2D array of shape (num_of_mc_runs, num_coeffs)
        """
        if self.num_params > 1 and (len(array.shape) != 2
                                    or array.shape[0] != self.num_params):
            raise ValueError("array must be of shape (num_params, num_of_mc_runs) -> "
                             f"({self.num_params}, num_of_mc_runs))")

        # check number of points greater than or equal to the number of coefficients
        if self.num_params == 1:
            num_of_mc_runs = array.shape[0]
        else:
            num_of_mc_runs = array.shape[1]

        minimum_num_coeff = self.num_coeffiencies(order)
        assert num_of_mc_runs >= minimum_num_coeff, \
            f"number of points ({num_of_mc_runs}) must be greater than or equal to " \
            f"the number of coefficients ({minimum_num_coeff})"

        # in 1D case, simply return the Vandermonde matrix
        if self.num_params == 1:
            return np.polynomial.polynomial.polyvander(array, order)

        num_params = self.num_params

        # for > 1D, enumeratively construct the Vandermonde matrix
        vander = [[0] * num_params]
        x = np.array(vander[0], dtype=np.int32)
        while True:
            if x[0] == order:
                break

            x = monomial.mono_next_grlex(num_params, x)
            vander.append(x.tolist())

        vander = np.array(vander, dtype=np.int32)

        # Take each parameter to powers of the monomial
        # array: [num_params, num_points]
        # vander: [num_coeffs, num_params]
        # array.T: [num_points, num_params]
        # vander[:, np.newaxis]: [num_coeffs, 1, num_params]
        V = jnp.power(array.T, vander[:, jnp.newaxis])
        V = jnp.prod(V, axis=-1).T
        return V

    def num_coeffiencies(self, order):
        """Return the number of coefficients for a polynomial of order `order`."""
        return monomial.mono_upto_enum(self.num_params, order)
