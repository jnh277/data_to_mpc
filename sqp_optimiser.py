
import numpy as np
import jax.numpy as jnp
from jax import device_put
from jax.scipy.special import expit

class SQP_NMPC_Solver:
    def __init__(self, simulate, state_constraints, input_constraints, N, Nx, Nu,):
        self.simulate = simulate
        self.state_constraints = state_constraints
        self.input_constraints = input_constraints
        self.N = N                                  # nmpc horizon
        self.Nx = Nx                                # state dimension (system order)
        self.Nu = Nu                                # input dimension
        self.ncu = len(input_constraints)           # number of input constraints
        self.ncx = len(state_constraints)           # number of state constraints
        self.n_eps = self.ncx * N                   # number of chance constraint slack variables needed
        self.nc = self.ncu + N * self.ncx           # total number of constraints

    @staticmethod
    def chance_constraint(hu, epsilon, gamma):  # Pr(h(u) >= 0 ) >= (1-epsilon)
        return jnp.mean(expit(hu / gamma), axis=1) - (1 - epsilon)  # take the sum over the samples (M)

    @staticmethod
    def __split_z(z, N, Nu, ncx):
        u = jnp.resshape(z[:Nu*N],(Nu, N))
        epsilon = z[Nu * N:N * Nu + ncx * N]  # slack variables on state constraints
        return u, epsilon

    # def constraints(self, x):
    #     x =


