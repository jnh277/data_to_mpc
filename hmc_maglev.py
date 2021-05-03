###############################################################################
#    Data to Controller for Nonlinear Systems: An Approximate Solution
#    Copyright (C) 2021  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
#    and James Holdsworth < james.holdsworth@newcastle.edu.au >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

""" This script runs simulation B) Rotary inverted pendulum from the paper and saves the results """
""" This script will take a fair amount of time to run if you have not installed cuda enabled JAX,
    presaved results are included and can be plotted without running this script """
""" The results can then be plotted using the script 'plot_pendulum_results.py' """

# general imports
import pystan
import numpy as np
from helpers import col_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.lax import scan
from jax.ops import index, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import solve_chance_logbarrier, log_barrier_cost

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy, or just do it 64-bit arith on CPU

# Control parameters
z_star = np.array([[0.008],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 13              # horizonline of MPC algorithm
sqc_v = np.array([10.,10.],dtype=float) # cost on state error
sqc = np.diag(sqc_v)
src = np.array([[10.]]) # cost on the input

# define the state constraints, (these need to be tuples)
position_upper_bound = 0.014 # m -- location of the post
position_lower_bound = 0.0 # m, location of the electromagnet
input_upper_bound = 1 # amps
input_lower_bound = -0.0001
state_constraints = (lambda z: position_lower_bound + z[[0],:,:],lambda z: -z[[0],:,:] + position_upper_bound)
input_constraints = (lambda u: input_upper_bound-u, lambda u: u + input_lower_bound)

# simulation parameters
# WARNING DONT MAKE T > 100 due to size of saved inv_metric
T = 2             # number of time steps to simulate and record measurements for
Ts = 0.004 # 250Hz ... 
z1_0 = 0.009
z2_0 = 0.0

r1_true = 0.0002        # measurement noise standard deviation
q1_true = 1e-6       # process noise standard deviation
q2_true = 1e-7       # process noise standard deviation

# got these values from the data sheet
Mb_true = 0.06 # kg, mass of the steel ball
Ldiff_true = 0.04 # H, difference between coil L at zero and infinity (L(0) - L(inf))
x50_true = 0.002 # m, location in mode of L where L is 50% of the infinity and 0 L's
grav = 9.81
k0_true = 0.5*x50_true*Ldiff_true/Mb_true
I0_true = x50_true
theta_true = {
        'Mb': Mb_true,
        'Ldiff': Ldiff_true,
        'x50': x50_true,
        'I0': I0_true,
        'k0': k0_true,
        'g': grav,
        'h': Ts
}
Nx = 2
Ny = 1
Nu = 1


## --------------------- FUNCTION DEFINITIONS ---------------------------------- ##
# ----------------- Simulate the system-------------------------------------------#
# def fill_theta(t): # unneeded
#     t['k0'] = 0.5 * t['x50'] * t['Ldiff'] / t['Mb']
#     t['I0'] = t['x50']
#     return t


def maglev_gradient(xt, u, t): # uses the lumped parametersation 
    dx = jnp.zeros_like(xt)
    dx = index_update(dx, index[0, :], xt[1, :])
    dx = index_update(dx, index[1, :], t['g'] - t['k0'] * u * u / (t['I0'] + xt[0,:]) / (t['I0'] + xt[0,:]))
    return dx

def current_current(xt,t):
    return np.sqrt(t['g'].mean()/t['k0'].mean())*(xt[0,:].mean() + t['I0'].mean())
def rk4(xt, ut, theta):
    h = theta['h']
    k1 = maglev_gradient(xt, ut, theta)
    k2 = maglev_gradient(xt + k1 * h / 2, ut, theta)
    k3 = maglev_gradient(xt + k2 * h / 2, ut, theta)
    k4 = maglev_gradient(xt + k3 * h, ut, theta)
    return xt + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * h  # should handle a 2D x just fine

def simulate(xt, u, w,
                  theta):  # w is expected to be 3D. xt is expected to be 2D. ut is expected to be 2d but also, should handle being a vector (3d)
    [Nx, Ns, Np1] = w.shape
    x = jnp.zeros((Nx, Ns, Np1 + 1))
    x = index_update(x, index[:, :, 0], xt)
    iis = jnp.arange(Np1)
    dict = {
        'x':x,
        'u':u,
        'w':w,
        'theta':theta
    }
    dict,_ = scan(scan_func,dict,iis) # carry, ys = scan(scan_func,dict,iis). for each ii in iis, runs dict,y = scan_func(dict,ii) and ys = ys.append(y)
    return dict['x'][:, :, 1:]  # return everything except xt

def scan_func(carry,ii):
    carry['x'] = index_update(carry['x'], index[:, :, ii + 1], rk4(carry['x'][:, :, ii], carry['u'][:, ii], carry['theta']) + carry['w'][:, :, ii])
    return carry, []

sim = jit(simulate)

# Pack true parameters
# theta_true = fill_theta(theta_true) 

# declare simulations variables
z_sim = np.zeros((Nx, 1, T+1), dtype=float) # state history
u = np.zeros((Nu, T+1), dtype=float)
w_sim = np.reshape(np.array([[q1_true],[q2_true]]),(Nx,1,1))*np.random.randn(Nx,1,T)
v = np.array([[r1_true]])*np.random.randn(Ny,T)
y = np.zeros((Ny, T), dtype=float)

# initial state
z_sim[0,:,0] = z1_0
z_sim[1,:,0] = z2_0


### hmc parameters and set up the hmc model
warmup = 300
chains = 4
iter = warmup + int(Ns/chains)
model_name = 'pendulum_diag'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

# load hmc warmup
# inv_metric = pickle.load(open('stan_traces/inv_metric.pkl','rb'))
# stepsize = pickle.load(open('stan_traces/step_size.pkl','rb'))
# last_pos = pickle.load(open('stan_traces/last_pos.pkl','rb'))


# define MPC cost, gradient and hessian function
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))


mu = 1
gamma = 1
delta = 0.05
max_iter = 5000

# declare some variables for storing the ongoing resutls
xt_est_save = np.zeros((Ns, Nx, T))
theta_est_save = np.zeros((Ns, 2, T))
q_est_save = np.zeros((Ns, Nx, T))
r_est_save = np.zeros((Ns, Ny, T))
uc_save = np.zeros((1, Nh, T))
mpc_result_save = []
hmc_traces_save = []

# predefine the mean of the prior so it doesnt change from run to run
theta_p_mu = np.array([1.0*I0_true, 1.0*k0_true])
u[0,0] = 0.

### SIMULATE SYSTEM AND PERFORM MPC CONTROL
    # simulate system
    # apply u_t
    # this takes x_t to x_{t+1}
    # measure y_t
    z_sim[:,:,[t+1]] = sim(z_sim[:,:,t],u[:,[t]],w_sim[:,:,[t]],theta_true)
    y[0, t] = z_sim[0, 0, t] + v[0, t]
    # y[2, t] = (u[0, t] - theta_true['Km'] * z_sim[2, 0, t]) / theta_true['Rm'] + v[2, t]


    # estimate system (estimates up to x_t)
    stan_data ={'no_obs': t+1,
                'Ts':Ts,
                'y': y[0, :t+1],
                'u': u[0, :t+1],
                'g':grav,
                'theta_p_mu':theta_p_mu,
                'theta_p_std':0.0005*np.array([I0_true, k0_true]),
                'r_p_mu': np.array([r1_true]),
                'r_p_std': 0.0005*np.array([r1_true]),
                'q_p_mu': np.array([q1_true, q2_true]),
                'q_p_std': 0.0005*np.array([q1_true, q2_true]),
                }


    fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains)
    traces = fit.extract()
    hmc_traces_save.append(traces)

    theta = traces['theta']
    h = traces['h']
    z = h[:,:,:t+1]

    r_mpc = traces['r']
    q_mpc = traces['q']

    # parameter samples
    I0_samps = theta[:, 0].squeeze()
    k0_samps = theta[:, 1].squeeze()

    theta_mpc = {
        'I0': I0_samps,
        'k0': k0_samps,
        'g': grav,
        'h': Ts
    }

    # current state samples (reshaped to [o,M])
    xt = z[:,:,-1].T
    # we also need to sample noise
    w_mpc = np.zeros((Nx, Ns, Nh + 1), dtype=float)
    w_mpc[0, :, :] = np.expand_dims(col_vec(q_mpc[:, 0]) * np.random.randn(Ns, Nh + 1),
                                    0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
    w_mpc[1, :, :] = np.expand_dims(col_vec(q_mpc[:, 1]) * np.random.randn(Ns, Nh + 1), 0)

    ut = u[:,[t]]  # control action that was just applied


    # save some things for later plotting
    xt_est_save[:, :, t] = z[:, :, -1]
    theta_est_save[:, :, t] = theta
    q_est_save[:, :, t] = q_mpc
    r_est_save[:, :, t] = r_mpc

    # calculate next control action
    if t > 0:
        uc0 = np.hstack((uc[[0],1:],np.reshape(uc[0,-1],(1,1))))
        mu = 200
        gamma = 0.2
        result = solve_chance_logbarrier(uc0, cost, gradient, hessian, ut, xt, theta_mpc, w_mpc, z_star, sqc,
                                         src,
                                         delta, simulate, state_constraints, input_constraints, verbose=2,
                                         max_iter=max_iter,mu=mu,gamma=gamma)
    else:
        cc = current_current(xt,theta_mpc):
        result = solve_chance_logbarrier(cc*np.ones((1, Nh)), cost, gradient, hessian, ut, xt, theta_mpc, w_mpc, z_star, sqc,
                                         src,
                                         delta, simulate, state_constraints, input_constraints, verbose=2,
                                         max_iter=max_iter)
    mpc_result_save.append(result)

    uc = result['uc']
    u[:,t+1] = uc[0,0]

    uc_save[0, :, t] = uc[0,:]

print(u)
print(z_sim[0,:])
fig = plt.figure(figsize=(6.4,4.8),dpi=300)
# plt.subplot(2, 2, 1)
plt.plot(np.arange(len(z_sim[0,:])),z_sim[0,:], color=u'#1f77b4',linewidth = 1,label='True')
# plt.axhline(position_upper_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
# plt.axhline(position_lower_bound, linestyle='--', color='r', linewidth=1.0)
# plt.xlim([0,49*0.025])
plt.ylabel('Ball position (m)')
plt.show()

# run = 'maglev_demo_results'
# with open('results/'+run+'/xt_est_save100.pkl','wb') as file:
#     pickle.dump(xt_est_save, file)
# with open('results/'+run+'/theta_est_save100.pkl','wb') as file:
#     pickle.dump(theta_est_save, file)
# with open('results/'+run+'/q_est_save100.pkl','wb') as file:
#     pickle.dump(q_est_save, file)
# with open('results/'+run+'/r_est_save100.pkl','wb') as file:
#     pickle.dump(r_est_save, file)
# with open('results/'+run+'/z_sim100.pkl','wb') as file:
#     pickle.dump(z_sim, file)
# with open('results/'+run+'/u100.pkl','wb') as file:
#     pickle.dump(u, file)
# with open('results/'+run+'/mpc_result_save100.pkl', 'wb') as file:
#     pickle.dump(mpc_result_save, file)
# with open('results/'+run+'/uc_save100.pkl', 'wb') as file:
#     pickle.dump(uc_save, file)
