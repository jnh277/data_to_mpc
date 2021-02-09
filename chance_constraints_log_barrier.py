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
No input constraints currently in place

Implementation:
Uses custom newton method to solve
Uses JAX to compile and run code on GPU/CPU and provide gradients and hessians
"""

# general imports
from numpy.core.numeric import zeros_like
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec
from pathlib import Path
import pickle
from scipy.optimize import minimize

# jax related imports
from jax.lax import dynamic_slice
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config
from jax.scipy.special import expit
config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy

second_order = True

#----------------- Parameters ---------------------------------------------------#

# Control parameters
z1_star = 1.0        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 20              # horizonline of MPC algorithm
qc1 = 1.0            # cost on state error
qc2 = 1.0
rc1 = 1.0             # cost on control action
x1_ub = 1.05         # upper bound constraint on state

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
z1_0 = 3.0            # initial x
z2_0 = 0.0
r1_true = 0.1        # measurement noise standard deviation
r2_true = 0.01
q1_true = 0.05       # process noise standard deviation
q2_true = 0.005       # process noise standard deviation
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
if second_order:
    # i should really avoid code duplication but i can feel it happening shortly
    print("2D MPC not done yet")
else:
    # jax compatible version of function to simulate forward a sample / scenario
    def simulate(xt, u, a, b, w_sim):
        N = len(u)
        M = len(a)
        x = jnp.zeros((M, N+1))
        # x[:, 0] = xt
        x = index_update(x, index[:,0], xt)
        for k in range(N):
            # x[:, k+1] = a * x[:, k] + b * u[k] + w_sim[:, k]
            x = index_update(x, index[:, k+1], a * x[:, k] + b * u[k] + w_sim[:, k])
        return x[:, 1:]

    #
    def logbarrier(z_sim, mu):       # log barrier for the constraint z_sim >= 0
        return jnp.sum(-mu * jnp.log(z_sim))

    def chance_constraint(x, s, x_ub, gamma, delta):    # upper bounded chance constraint on the state
        return jnp.mean(expit((x_ub - x) / gamma), axis=0) - (1 - s)     # take the sum over the samples (M)

    # jax compatible version of function to compute cost
    def cost(z_sim, ut, xt, x_star, a, b, w_sim, qc, rc, mu, gamma, x_ub, delta, N):
        # uc is given by z_sim[:(N-1)]
        # epsilon is given by z_sim[(N-1):2(N-1)]
        # other slack variables could go after this

        uc = z_sim[:(N-1)]              # control input variables  #
        epsilon = z_sim[19:]               # slack variables

        u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is N-1
        x = simulate(xt, u, a, b, w_sim)
        # state error and input penalty cost and cost that drives slack variables down
        V1 = jnp.sum((qc*(x - x_star)) ** 2) + jnp.sum((rc * uc)**2) + jnp.sum(10 * (epsilon + 1e3)**2)
        # need a log barrier on each of the slack variables to ensure they are positve
        V2 = logbarrier(epsilon - delta, mu)       # aiming for 1-delta% accuracy
        # now the chance constraints
        cx = chance_constraint(x[:,1:], epsilon, x_ub, gamma, delta)
        V3 = logbarrier(cx, mu)
        return V1 + V2 + V3

    # compile cost and create gradient and hessian functions
    cost_jit = jit(cost, static_argnums=(13,))  # static argnums means it will recompile if N changes
    gradient = jit(grad(cost, argnums=0), static_argnums=(13,))    # get compiled function to return gradients with respect to z_sim (uc, s)
    hessian = jit(jacfwd(jacrev(cost, argnums=0)), static_argnums=(13,))

    # define some optimisation settings
    mu = 1e4
    gamma = 1
    x_ub = 1
    delta = 0.01

    # put everything we want to call onto the gpu
    args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
            device_put(b), device_put(w_sim), device_put(qc), device_put(rc),
            device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta))

    # test
    z0 = jnp.hstack([jnp.zeros((N-1)), np.ones((N-1,))])
    test = cost_jit(z0, *args, N)
    testgrad = gradient(z0, *args, N)
    testhessian = hessian(z0, *args, N)

    max_iter = 100

    z_sim = np.hstack([np.zeros((N-1,)), np.ones((N-1,))]) 
    mu = 1e4
    gamma = 1.0
    args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
            device_put(b), device_put(w_sim), device_put(qc), device_put(rc),
            device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta))

    for i in range(max_iter):
        # compute cost, gradient, and hessian
        jz = device_put(z_sim)
        c = np.array(cost_jit(jz, *args, N))
        g = np.array(gradient(jz, *args, N))
        h = np.array(hessian(jz, *args, N))

        # compute search direction
        p = - np.linalg.solve(h, g)
        # check that we have a valid search direction and if not then fix
        # TODO: make this less hacky (look at slides)
        beta2 = 1e-8
        while np.dot(p,g) > 0:
            p = - np.linalg.solve((h + beta2 * np.eye(h.shape[0])),g)
            beta2 = beta2 * 2

        # perform line search
        alpha = 1.0
        for k in range(52):
            # todo: (lower priority) need to use the wolfe conditions to ensure a bit of a better decrease
            ztest = z_sim + alpha * p
            ctest = np.array(cost_jit(device_put(ztest), *args, N))
            # if np.isnan(ctest) or np.isinf(ctest):
            #     continue
            # nan and inf checks should be redundant
            if ctest < c:
                z_sim = ztest
                break

            alpha = alpha / 2

        if k == 51:
            print('Failed line search')
            break

        print('Iter:', i+1, 'Cost: ', c, 'nd:',np.dot(g,p),'alpha: ', alpha, 'mu: ', mu, 'gamma: ', gamma)

        if np.abs(np.dot(g,p)) < 1e-2: # if search direction was really small, then decrease mu and s for next iteration
            if mu < 1e-6 and gamma < 1e-3:
                break   # termination criteria satisfied

            mu = max(mu / 2, 0.999e-6)
            gamma = max(gamma / 1.25, 0.999e-3)
            # need to adjust the slack after changing gamma
            x_new = simulate(xt, np.hstack([ut, z_sim[:N-1]]), a, b, w_sim)
            cx = chance_constraint(x_new[:, 1:], z_sim[N-1:], x_ub, gamma, delta)
            tmp = -np.minimum(cx, 0)+1e-5
            z_sim[N-1:] += -np.minimum(cx, 0)+1e-6
            args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
                    device_put(b), device_put(w_sim), device_put(qc), device_put(rc),
                    device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta))


    x_mpc = simulate(xt, np.hstack([ut, z_sim[:N-1]]), a, b, w_sim)
    cx = chance_constraint(x_mpc, 0.0, x_ub, gamma, delta)
    print('Constraint satisfaction')
    print(1 + cx)
    #
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.hist(x_mpc[:, i*3], label='MC forward sim')
        if i==1:
            plt.title('MPC solution over horizon')
        plt.axvline(1.0, linestyle='--', color='g', linewidth=2, label='target')
        plt.xlabel('t+'+str(i*3+1))
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.plot(z_sim[:N-1])
    plt.title('MPC determined control action')
    plt.show()

print("Finito")





