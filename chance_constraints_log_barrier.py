"""
Simulate a first order state space model given by
x_{t+1} = a*x_t + b*u_t + w_t
y_t = x_t + e_t
with q and r the standard deviations of w_t and q_t respectively

Jointly estimate the smoothed state trajectories and parameters theta = {a, b, q, r}
to give p(x_{1:T}, theta | y_{1:T})

GOAL:
Use a Monte Carlo style approach to perform MPC where the state constraints are satisfied
with a given probability.

Current set up: Uses MC to give an expected cost and then satisfies,
Chance state constraints using a log barrier formulation
Chance state constraints using a log barrier formulation
Input constraints using a log barrier formulation

Implementation:
Uses custom newton method to solve
Uses JAX to compile and run code on GPU/CPU and provide gradients and hessians
"""

## TODO list
# refactor the parameters a little bit to put them into a dictionary, so that the same function can be used for the ssm simulator and the jax mpc simulator


# general imports
from numpy.core.numeric import zeros_like
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec
from pathlib import Path
import pickle

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy

second_order = True

#----------------- Parameters ---------------------------------------------------#

# Control parameters
z_star = np.array([[1.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 20              # horizonline of MPC algorithm
sqc_v = np.array([1.0,1.0],dtype=float)            # cost on state error
sqc = np.diag(sqc_v)
src_v = np.array([1.0,1.0],dtype=float)  
src = np.diag(src_v)            # cost on control action

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
z1_0 = 3.0            # initial x
z2_0 = 0.0
r1_true = 0.1        # measurement noise standard deviation
r2_true = 0.01
q1_true = 0.05       # process noise standard deviation
q2_true = 0.005      # process noise standard deviation
m_true = 1;
b_true = 0.7
k_true = 0.25

Nx = 2;
Ny = 2;
Nu = 1;

#----------------- Simulate the system-------------------------------------------#

# def ssm1(x1, x2, u, T, a11=0.0, a12=1.0, b1=0.0):
#     return (a11*x1 + a12*x2 + b1*u)*T
# def ssm2(x1, x2, u, T, a21=0.5, a22=-0.5, b2=0.3):
#     return a21*x1 + a22*x2 + b2*u
def ssm_euler(x,u,A,B,T):
    return (np.matmul(A,x) + np.matmul(B,u)) * T;

# SSM equations
A = np.zeros((Nx,Nx), dtype=float)
B = np.zeros((Nx,Nu), dtype=float)

A[0,1] = 1.0;
A[1,0] = -k_true/m_true
A[1,1] = -b_true/m_true
B[1,0] = 1/m_true;


z_sim = np.zeros((Nx,T+1), dtype=float) # state history

# initial state
z_sim[0,0] = z1_0 
z_sim[1,0] = z2_0 

# noise predrawn and independant
w_sim = np.zeros((Nx,T),dtype=float)
w_sim[0,:] = np.random.normal(0.0, q1_true, T)
w_sim[1,:] = np.random.normal(0.0, q2_true, T)

# create some inputs that are random but held for 10 time steps
u = np.random.uniform(-0.5,0.5, T*Nu)
u = np.reshape(u, (Nu,T))

# spicy
for k in range(T):
    # x1[k+1] = ssm1(x1[k],x2[k],u[k]) + w1[k]
    # x2[k+1] = ssm2(x1[k],x2[k],u[k]) + w2[k]
    z_sim[:,k+1] = ssm_euler(z_sim[:,k],u[:,k],A,B,1.0) + w_sim[:,k]

# simulate measurements
v = np.zeros((Ny,T), dtype=float)
v[0,:] = np.random.normal(0.0, r1_true, T)
v[1,:] = np.random.normal(0.0, r2_true, T)
y = np.zeros((Ny,T), dtype=float)
y[0,:] = z_sim[0,:-1]
y[1,:] = (-k_true*z_sim[0,:-1] -b_true*z_sim[1,:-1] + u[0,:])/m_true
y = y + v; # add noise

plt.subplot(2,1,1)
plt.plot(u[0,:])
plt.plot(y[1,:],linestyle='None',color='r',marker='*')
plt.title('Simulated inputs and measurement used for inference')
plt.subplot(2, 1, 2)
plt.plot(z_sim[0,:])
plt.plot(y[0,:],linestyle='None',color='r',marker='*')
plt.title('Simulated state 1 and measurements used for inferences')
plt.tight_layout()
plt.show()

#----------- USE HMC TO PERFORM INFERENCE ---------------------------#
# avoid recompiling
model_name = 'LSSM_O2_MSD_demo'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

stan_data = {
    'N':T,
    'y':y,
    'u':u,
    'O':Nx,
    'D':Ny,
    'T':1.0
}

fit = model.sampling(data=stan_data, warmup=1000, iter=2000)
traces = fit.extract()

# state samples
z_samps = np.transpose(traces['z'],(1,0,2)) # Ns, Nx, T --> Nx, Ns, T


# parameter samples
m_samps = traces['m'].squeeze()
k_samps = traces['k'].squeeze()
b_samps = traces['b'].squeeze() # single valued parameters shall 1D numpy objects! The squeeze has been squoze
q_samps = np.transpose(traces['q'],(1,0)) 
r_samps = np.transpose(traces['r'],(1,0))

# plot the initial parameter marginal estimates
q1plt = q_samps[0,:].squeeze()
q2plt = q_samps[1,:].squeeze()
r1plt = r_samps[0,:].squeeze()
r2plt = r_samps[1,:].squeeze()


plot_trace(m_samps,2,4,1,'m')
plt.title('HMC inferred parameters')
plot_trace(k_samps,2,4,2,'k')
plot_trace(b_samps,2,4,3,'b')
plot_trace(q1plt,2,4,4,'q1')
plot_trace(q2plt,2,4,5,'q2')
plot_trace(r1plt,2,4,6,'r1')
plot_trace(r2plt,2,4,7,'r2')
plt.show()

# plot some of the initial marginal state estimates
for i in range(4):
    if i==1:
        plt.title('HMC inferred position')
    plt.subplot(2,2,i+1)
    plt.hist(z_samps[0,:,i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
    plt.axvline(z_sim[0,i*20+1], label='True', linestyle='--',color='k',linewidth=2)
    plt.xlabel('x_'+str(i+1))
plt.tight_layout()
plt.legend()
plt.show()


# downsample the the HMC output since for illustration purposes we sampled > M
ind = np.random.choice(len(m_samps), Ns, replace=False)
m_mpc = m_samps[ind]  # same indices for all to ensure they correpond to the same realisation from dist
k_mpc = k_samps[ind]
b_mpc = b_samps[ind]
q_mpc = q_samps[:,ind]
r_mpc = r_samps[:,ind]
zt = z_samps[:,ind,-1]  # inferred state for current time step

# predraw noise from sampled q! --> Ns number of Nh long scenarios assocation with a particular q
w_mpc = np.zeros((Nx,Ns,Nh+1),dtype=float)
w_mpc[0,:,:] = np.expand_dims(col_vec(q_mpc[0,:]) * np.random.randn(Ns, Nh+1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
w_mpc[1,:,:] = np.expand_dims(col_vec(q_mpc[1,:]) * np.random.randn(Ns, Nh+1), 0)
ut = np.expand_dims(u[:,-1], axis=1)      # control action that was just applied

# At this point I have:
# zt which is the current inferred state, and is [Nx,Ns]: Ns samples of Nx column vector
# ut which is the previously actioned control signal, and is [Nu,1]: 2D because of matmul
# m,k,b *_mpc are [Ns] arrays of parameter value samples: 1D because simplicity
# q,r *_mpc are [Nx/y,Ns]. Don't know why!

# ----- Solve the MC MPC control problem ------------------#

# jax compatible version of function to simulate forward the samples / scenarios
def simulate(xt, u, w, theta):
    a = theta['a']
    b = theta['b']
    [o, M, N] = w.shape
    x = jnp.zeros((o, M, N+1))
    # x[:, 0] = xt
    x = index_update(x, index[:, :,0], xt)
    for k in range(N):
        # x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
        x = index_update(x, index[:, :, k+1], a * x[:, :, k] + b * u[:, k] + w[:, :, k])
    return x[:, :, 1:]

def msd_simulate(xt, u, w, theta):
    m = theta['m']
    k = theta['k']
    b = theta['b']
    tau = theta['tau']
    [Nx, Ns, Nh] = w.shape
    x = jnp.zeros((Nx, Ns, Nh))
    # x[:, 0] = xt
    x = index_update(x, index[:, :, 0], xt)
    for ii in range(Nh):
        # x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
        x = index_update(x, index[0, :, k+1], tau*x[1, :, k])
        x = index_update(x, index[1, :, k+1], tau*(-k*(1/m)*x[0, :, k] -b*(1/m)*x[1, :, k] + (1/m)*u[:,k]))

    return x[:, :, 1:]


# compile cost and create gradient and hessian functions
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))


# define some optimisation settings
mu = 1e4
gamma = 1
delta = 0.05
# delta = np.linspace(0.05,0.2,N)     # varying the requried probabilistic satisfaction
max_iter = 1000


# define some state constraints, (these need to be tuples (so trailing comma))
z_ub = jnp.array([[1.2],[jnp.Inf]])
state_constraints = (lambda x: z_ub - z,)
# state_constraints = ()
input_constraints = (lambda u: 5.0 - u,)
# input_constraints = ()

theta = {'m':m_samps,
         'k':k_samps,
         'b':b_samps,
         'tau': 1.0
         }


# solve mpc optimisation problem
result = solve_chance_logbarrier(np.zeros((1,N)), cost, gradient, hessian, ut, zt, theta, w_mpc, z_star, sqc, src,
                            delta, simulate, state_constraints, input_constraints)

uc = result['uc']

# if len(state_constraints) > 0:
#     x_mpc = simulate(xt, np.hstack([ut, uc]), w, theta)
#     hx = np.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
#     cx = np.mean(hx > 0, axis=1)
#     print('State constraint satisfaction')
#     print(cx)
# if len(input_constraints) > 0:
#     cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints],axis=1)
#     print('Input constraint satisfaction')
#     print(cu >= 0)
# #
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.hist(x_mpc[0,:, i*3], label='MC forward sim')
#     if i==1:
#         plt.title('MPC solution over horizon')
#     plt.axvline(x_star, linestyle='--', color='g', linewidth=2, label='target')
#     plt.axvline(x_ub, linestyle='--', color='r', linewidth=2, label='upper bound')
#     plt.xlabel('t+'+str(i*3+1))
# plt.tight_layout()
# plt.legend()
# plt.show()

# plt.plot(uc[0,:])
# plt.title('MPC determined control action')
# plt.show()







