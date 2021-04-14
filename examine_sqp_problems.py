# fixing multithreading on osx
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")

# general imports
import pystan
import numpy as np
from helpers import col_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm
import quadprog
import time
import matplotlib.pyplot as plt

# jax related imports
from jax.scipy.special import expit
import jax.numpy as jnp
from jax import grad, jit,  jacfwd, jacrev
from jax.ops import index, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy



with open('results/sqp_debug/saved_optims.pkl','rb') as file:
    saved_optims = pickle.load(file)


# Overview of final state of each mpc problem
iters_taken = np.hstack([prob['iters'] for prob in saved_optims])
final_norm_gradL = np.hstack([np.sum(prob['grad L']**2) for prob in saved_optims])
final_nd = np.hstack([prob['nd'] for prob in saved_optims])
min_final_C = np.hstack([np.min(prob['C']) for prob in saved_optims])
max_final_cx = np.hstack([np.max(prob['cx']) for prob in saved_optims])
final_cost = np.hstack([prob['cost'] for prob in saved_optims])

plt.subplot(4,1,1)
plt.plot(iters_taken+1)
plt.title('Iterations taken, max allowed was 30')

plt.subplot(4,1,2)
plt.plot(final_norm_gradL > 1)
plt.title('Final norm grad L > 1.0')

plt.subplot(4,1,3)
plt.plot(np.abs(final_nd)>1e-2)
plt.title('Final |nd| > 1e-2')

plt.subplot(4,1,4)
plt.plot(min_final_C)
plt.title('Min constraint value (constraints are of form C>0)')
plt.tight_layout()
plt.show()


plt.subplot(3,1,1)
plt.plot(iters_taken+1)
plt.title('Iterations taken, max allowed was 30')

plt.subplot(3,1,2)
plt.plot(max_final_cx > 0.05001)
plt.title('max final % state violation > 0.05% (desired)')

plt.subplot(3,1,3)
plt.plot(final_cost)
plt.title('final_cost')

plt.tight_layout()
plt.show()


# look at info from subproblems when solving problem number 3 or 5
# subproblems = saved_optims[3]['subproblems']
subproblems = saved_optims[5]['subproblems']

norm_gradL = np.hstack([np.sum(subprob['gradL']**2) for subprob in subproblems])
gamma = np.hstack([subprob['gamma'] for subprob in subproblems])
min_C = np.hstack([np.min(subprob['C']) for subprob in subproblems])
cost = np.hstack([subprob['cost'] for subprob in subproblems])
max_cx = np.hstack([np.max(subprob['cx']) for subprob in subproblems])
alpha = np.hstack([subprob['alpha'] for subprob in subproblems])
nd = np.hstack([subprob['nd'] for subprob in subproblems])
ind_max_cx = np.hstack([np.argmax(subprob['cx']) for subprob in subproblems])
deriv_max_cx = np.hstack([np.max(np.min(subprob['dC'][ind_max_cx[i],:10])) for i,subprob in enumerate(subproblems)])
pdotdxc = np.hstack([subprob['p'].dot(subprob['dC'][ind_max_cx[i],:]) for i,subprob in enumerate(subproblems)])
min_eig_H = np.hstack([np.min(np.linalg.eig(subprob['H'])[0]) for subprob in subproblems])

plt.subplot(5,1,1)
plt.semilogy(gamma)
plt.title('Gamma (min value is 0.001 and starts at 10)')

plt.subplot(5,1,2)
plt.semilogy(norm_gradL)
plt.title('norm gradL')

plt.subplot(5,1,3)
plt.plot(min_C > -1e-6)
plt.title('min C value > -1e-6 (exit cond for constraints of form C > 0)')

plt.subplot(5,1,4)
plt.semilogy(cost)
plt.title('cost')

plt.subplot(5,1,5)
plt.semilogy(np.abs(nd))
plt.title('|nd|')

plt.tight_layout()
plt.show()


plt.subplot(5,1,1)
plt.semilogy(gamma)
plt.title('Gamma (min value is 0.001 and starts at 10)')

plt.subplot(5,1,2)
plt.plot(max_cx > 0.05001)
plt.title('max % state viol > 0.05% (desired)')

plt.subplot(5,1,3)
plt.plot(np.abs(deriv_max_cx))
plt.title('gradient worst cx ')

plt.subplot(5,1,4)
plt.plot(np.abs(pdotdxc))
plt.title('gradient worst cx dot product with search direction')

plt.subplot(5,1,5)
plt.plot(min_eig_H > 1e-6)
plt.title('H is PD')

plt.tight_layout()
plt.show()