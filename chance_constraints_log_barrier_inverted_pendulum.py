"""
Simulate a nonlinear rotarary inverted pendulum (QUBE SERVO2 SYSTEM)
with independent process and measurement noise

Jointly estimate the smoothed state trajectories and parameters
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
import math

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy



#----------------- Parameters ---------------------------------------------------#

# Control parameters
z_star = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 25              # horizonline of MPC algorithm
sqc_v = np.array([1,10.0,1e-5,1e-5],dtype=float)            # cost on state error
sqc = np.diag(sqc_v)
src = np.array([[0.001]])

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
Ts = 0.025
z1_0 = -np.pi/4            # initial states
z2_0 = np.pi/4
z3_0 = 0.0
z4_0 = 0.0

# got these values from running HMC on some real data
r1_true = 0.0011        # measurement noise standard deviation
r2_true = 0.001
r3_true = 0.175
q1_true = 3e-4       # process noise standard deviation
q2_true = 1e-4      # process noise standard deviation
q3_true = 0.013       # process noise standard deviation
q4_true = 0.013      # process noise standard deviation

# got these values from the data sheet
mr_true = 0.095 # kg
mp_true = 0.024 # kg
Lp_true = 0.129 # m
Lr_true = 0.085 # m
Jr_true = mr_true * Lr_true * Lr_true / 3 # kgm^2
Jp_true = mp_true * Lp_true * Lp_true / 3 # kgm^2
Km_true = 0.042 # Vs/rad / Nm/A
Rm_true = 8.4 # ohms
Dp_true = 5e-5 # Nms/rad
Dr_true = 1e-3 # Nms/rad
grav = 9.81

theta_true = {
        'Mp': mp_true,
        'Lp': Lp_true,
        'Lr': Lr_true,
        'Jr': Jr_true,
        'Jp': Jp_true,
        'Km': Km_true,
        'Rm': Rm_true,
        'Dp': Dp_true,
        'Dr': Dr_true,
        'g': grav,
        'h': Ts
    }
Nx = 4
Ny = 3
Nu = 1

#----------------- Simulate the system-------------------------------------------#

def fill_theta(t):
    Jr = t['Jr']
    Jp = t['Jp']
    Mp = t['Mp']
    Lr = t['Lr']
    Lp = t['Lp']
    g = t['g']

    t['Jr + Mp * Lr * Lr'] = Jr + Mp * Lr * Lr
    t['0.25 * Mp * Lp * Lp'] = 0.25 * Mp * Lp * Lp
    t['0.5 * Mp * Lp * Lr'] = 0.5 * Mp * Lp * Lr
    t['m22'] = Jp + 0.25 * Mp * Lp * Lp
    t['0.5 * Mp * Lp * g'] = 0.5 * Mp * Lp * g
    return t

def qube_gradient(xt,u,t): # t is theta, this is for QUBE
    cos_xt1 = jnp.cos(xt[1,:]) # there are 5 of these
    sin_xt1 = jnp.sin(xt[1,:]) # there are 4 of these
    m11 = t['Jr + Mp * Lr * Lr'] + t['0.25 * Mp * Lp * Lp'] - t['0.25 * Mp * Lp * Lp'] * cos_xt1 * cos_xt1
    m12 = t['0.5 * Mp * Lp * Lr'] * cos_xt1
    m22 = t['m22'] # this should be a scalar anyway - can be vector
    sc = m11 * m22 - m12 * m12
    tau = (t['Km'] * (u[0] - t['Km'] * xt[2,:])) / t['Rm'] # u is a scalr
    d1 = tau - t['Dr'] * xt[2,:] - 2 * t['0.25 * Mp * Lp * Lp'] * sin_xt1 * cos_xt1 * xt[2,:] * xt[3,:] + t['0.5 * Mp * Lp * Lr'] * sin_xt1 * xt[3,:] * xt[3,:]
    d2 = -t['Dp'] * xt[3,:] + t['0.25 * Mp * Lp * Lp'] * cos_xt1 * sin_xt1 * xt[2,:] * xt[2,:] - t['0.5 * Mp * Lp * g'] * sin_xt1
    dx = jnp.zeros_like(xt)
    dx = index_update(dx, index[0, :], xt[2,:])
    dx = index_update(dx, index[1, :], xt[3,:])
    dx = index_update(dx, index[2, :], (m22 * d1 - m12 * d2)/sc)
    dx = index_update(dx, index[3, :], (m11 * d2 - m12 * d1)/sc)
    return dx
    
def rk4(xt,ut,theta):
    h = theta['h']
    k1 = qube_gradient(xt,ut,theta)
    k2 = qube_gradient(xt + k1*h/2,ut,theta)
    k3 = qube_gradient(xt + k2*h/2,ut,theta)
    k4 = qube_gradient(xt + k3*h,ut,theta)
    return xt + (k1/6 + k2/3 + k3/3 + k4/6)*h # should handle a 2D x just fine

def pend_simulate(xt,u,w,theta):# w is expected to be 3D. xt is expected to be 2D. ut is expected to be 2d but also, should handle being a vector (3d)
    [Nx,Ns,Np1] = w.shape
    # if u.ndim == 2:  # conditional checks might slow jax down
    #     Nu = u.shape[1]
        # if Nu != Np1:
        #     print('wyd??')
    x = jnp.zeros((Nx, Ns, Np1+1))
    x = index_update(x, index[:, :, 0], xt)
    for ii in range(Np1):
        x = index_update(x, index[:, :, ii+1], rk4(x[:,:,ii],u[:,ii],theta) + w[:, :, ii]) # slicing creates 2d x into 3d x. Also, Np1 loop will consume all of w
    return x[:, :, 1:]  # return everything except xt 


# compile cost and create gradient and hessian functions
sim = jit(pend_simulate)  # static argnums means it will recompile if N changes

# THETA WILL HAVE TO CONTAIN SOME REALLY WEIRD SHIT!
theta_true = fill_theta(theta_true)

z_sim = np.zeros((Nx,1,T+1), dtype=float) # state history

# initial state
z_sim[0,:,0] = z1_0 
z_sim[1,:,0] = z2_0 
z_sim[2,:,0] = z3_0
z_sim[3,:,0] = z4_0 

# noise predrawn and independant
w_sim = np.reshape(np.array([[0.],[q2_true],[q3_true],[q4_true]]),(4,1,1))*np.random.randn(4,1,T)


# create some inputs that are random but held for 10 time steps
u = np.zeros((Nu, T+1), dtype=float)
for i in range(int(T/10)):
    u[0,i*10:] = np.random.uniform(-18,18,(1,))

# spicy
for k in range(T):
    z_sim[:,:,[k+1]] = sim(z_sim[:,:,k],u[:,[k]],w_sim[:,:,[k]],theta_true)

# simulate measurements
v = np.array([[r1_true],[r2_true],[r3_true]])*np.random.randn(3,T)
y = np.zeros((3,T), dtype=float)
y[0:2,:] = z_sim[0:2,0,:-1]
y[2,:] = (u[0,:] - theta_true['Km'] * z_sim[2,0,:-1]) / theta_true['Rm']
y = y + v           # add noise

plt.subplot(2,1,1)
plt.plot(u[0,:])
# plt.plot(y[1,:],linestyle='None',color='r',marker='*')
plt.title('Simulated inputs and measurement used for inference')
plt.subplot(2, 1, 2)
plt.plot(z_sim[0,0,:])
plt.plot(y[0,:],linestyle='None',color='r',marker='*')
plt.title('Simulated state 1 and measurements used for inferences')
plt.tight_layout()
plt.show()

#----------- USE HMC TO PERFORM INFERENCE ---------------------------#

trace_name = 'inverted_pendulum_trace'
trace_path = 'stan_traces/'
init_name = 'inverted_pendulum_init'
dont_stan = False


if Path(trace_path+trace_name+'.pkl').is_file() & Path(trace_path+init_name+'.pkl').is_file() & dont_stan:
    # avoid redoing HMC
    z_init = pickle.load(open(trace_path+init_name+'.pkl','rb'))
    traces = pickle.load(open(trace_path+trace_name+'.pkl', 'rb'))
    no_obs = y.shape[1]

else:
    # avoid recompiling
    model_name = 'pendulum_diag'
    path = 'stan/'
    if Path(path+model_name+'.pkl').is_file():
        model = pickle.load(open(path+model_name+'.pkl', 'rb'))
    else:
        model = pystan.StanModel(file=path+model_name+'.stan')
        with open(path+model_name+'.pkl', 'wb') as file:
            pickle.dump(model, file)
    no_obs = y.shape[1]
    stan_data ={'no_obs': no_obs,
                'Ts':Ts,
                'y': y,
                'u': u.flatten(),
                'Lr':Lr_true,
                'Mp':mp_true,
                'Lp':Lp_true,
                'g':grav,
                'theta_p_mu':np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true]),
                # 'theta_p_std':0.5*np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true]),
                'theta_p_std':0.1*np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true]),
                'r_p_mu': np.array([r1_true, r2_true, r3_true]),
                'r_p_std': 0.5*np.array([r1_true, r2_true, r3_true]),
                'q_p_mu': np.array([q1_true, q2_true, q3_true, q4_true]),
                'q_p_std': np.array([q1_true, q2_true, 0.5*q3_true, 0.5*q3_true]),
                }

    control = {"adapt_delta": 0.85,
            "max_treedepth":13}         # increasing from default 0.8 to reduce divergent steps

    # Determine initialisation for the chains
    # state initialisation point (initlaise based off measurements)
    z_init = np.zeros((4,no_obs+1))
    z_init[0,:-1] = y[0,:]
    z_init[1,:-1] = y[1,:]
    z_init[0,-1] = y[0,-1]      # repeat last entry
    z_init[1,-1] = y[1,-1]      # repeat last entry
    z_init[2,:-1] = np.gradient(y[0,:])/Ts
    z_init[2,-1] = z_init[2,-2]
    z_init[3,:-1] = np.gradient(y[1,:])/Ts
    z_init[3,-1] = z_init[3,-2]

    # parameter chain initialisation (going to start at true values +/- 20 percent)
    theta0 = np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true])

    def init_function():
        output = dict(theta=theta0.flatten() * np.random.uniform(0.8,1.2,np.shape(theta0.flatten())),
                    h=z_init + np.random.normal(0.0,0.1,np.shape(z_init)),
                    )
        return output


    fit = model.sampling(data=stan_data, warmup=2000, iter=2200, chains=4,control=control, init=init_function)

    traces = fit.extract()
    with open(trace_path+trace_name+'.pkl', 'wb') as file:
        pickle.dump(traces, file)
    with open(trace_path+init_name+'.pkl','wb') as file:
        pickle.dump(z_init,file)

    # stepsize = fit.get_stepsize()
    # inv_metric = fit.get_inv_metric(as_dict=True)
    # last_pos = fit.get_last_position()
    # with open('stan_traces/step_size.pkl','wb') as file:
    #     pickle.dump(stepsize,file)
    # with open('stan_traces/inv_metric.pkl', 'wb') as file:
    #     pickle.dump(inv_metric, file)
    # with open('stan_traces/last_pos.pkl', 'wb') as file:
    #     pickle.dump(last_pos, file)
    

theta = traces['theta']
z = traces['h'][:,:,:no_obs]
theta_mean = np.mean(theta,0)
z_mean = np.mean(z,0)

plot_trace(theta[:,0],3,1,'Jr')
plot_trace(theta[:,1],3,2,'Jp')
plot_trace(theta[:,2],3,3,'Km')
plt.show()

plot_trace(theta[:,3],3,1,'Rm')
plot_trace(theta[:,4],3,2,'Dp')
plot_trace(theta[:,5],3,3,'Dr')
plt.show()

plt.subplot(2,2,1)
plt.plot(y[0,:])
plt.plot(z_mean[0,:])
plt.xlabel('time')
plt.ylabel(r'arm angle $\theta$')
plt.legend(['Measurements','mean estimate'])

plt.subplot(2,2,2)
plt.plot(y[1,:])
plt.plot(z_mean[1,:])
plt.xlabel('time')
plt.ylabel(r'pendulum angle $\alpha$')
plt.legend(['Measurements','mean estimate'])

plt.subplot(2,2,3)
plt.plot(z_init[2,:])
plt.plot(z_mean[2,:])
plt.xlabel('time')
plt.ylabel(r'arm angular velocity $\dot{\theta}$')
plt.legend(['Grad measurements','mean estimate'])

plt.subplot(2,2,4)
plt.plot(z_init[3,:])
plt.plot(z_mean[3,:])
plt.xlabel('time')
plt.ylabel(r'pendulum angular velocity $\dot{\alpha}$')
plt.legend(['Grad measurements','mean estimate'])
plt.show()


# state samples
z_samps = np.transpose(z,(1,0,2)) # Ns, Nx, T --> Nx, Ns, T

# parameter samples
Jr_samps = theta[:,0].squeeze()
Jp_samps = theta[:,1].squeeze()
Km_samps = theta[:,2].squeeze()
Rm_samps = theta[:,3].squeeze()
Dp_samps = theta[:,4].squeeze()
Dr_samps = theta[:,5].squeeze()
q_samps = np.transpose(traces['q'],(1,0))
r_samps = np.transpose(traces['r'],(1,0))
#
# downsample the the HMC output since for illustration purposes we sampled > M
ind = np.random.choice(len(Jr_samps), Ns, replace=False)
theta_mpc = {
        'Mp': mp_true,
        'Lp': Lp_true,
        'Lr': Lr_true,
        'Jr': Jr_samps[ind],
        'Jp': Jp_samps[ind],
        'Km': Km_samps[ind],
        'Rm': Rm_samps[ind],
        'Dp': Dp_samps[ind],
        'Dr': Dr_samps[ind],
        'g': grav,
        'h': Ts
}

theta_mpc = fill_theta(theta_mpc)

q_mpc = q_samps[:,ind]
r_mpc = r_samps[:,ind]
zt = z_samps[:,ind,-1]  # inferred state for current time step
#
# predraw noise from sampled q! --> Ns number of Nh long scenarios assocation with a particular q
w_mpc = np.zeros((Nx,Ns,Nh+1),dtype=float)
w_mpc[0,:,:] = np.expand_dims(col_vec(q_mpc[0,:]) * np.random.randn(Ns, Nh+1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
w_mpc[1,:,:] = np.expand_dims(col_vec(q_mpc[1,:]) * np.random.randn(Ns, Nh+1), 0)
w_mpc[2,:,:] = np.expand_dims(col_vec(q_mpc[2,:]) * np.random.randn(Ns, Nh+1), 0)
w_mpc[3,:,:] = np.expand_dims(col_vec(q_mpc[3,:]) * np.random.randn(Ns, Nh+1), 0)
# ut = u[:,-1]
ut = np.expand_dims(u[:,-1], axis=1)      # control action that was just applied
#
# # At this point I have:
# # zt which is the current inferred state, and is [Nx,Ns]: Ns samples of Nx column vector
# # ut which is the previously actioned control signal, and is [Nu,1]: 2D because of matmul
# # m,k,b *_mpc are [Ns] arrays of parameter value samples: 1D because simplicity
# # q,r *_mpc are [Nx/y,Ns]. Don't know why!

# # ----- Solve the MC MPC control problem ------------------#
# # jax compatible version of function to simulate forward the samples / scenarios
# compile cost and create gradient and hessian functions
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))
#
#
# define some optimisation settings
mu = 1e4
gamma = 1
delta = 0.05
max_iter = 10000


# define some state constraints, (these need to be tuples (so trailing comma))
state_constraints = (lambda z: 0.75*np.pi - z[[0],:,:],lambda z: z[[0],:,:] + 0.75*np.pi)

# define some input constraints
input_constraints = (lambda u: 18. - u, lambda u: u + 18.)

# # solve mpc optimisation problem
result = solve_chance_logbarrier(np.zeros((1,Nh)), cost, gradient, hessian, ut, zt, theta_mpc, w_mpc, z_star, sqc, src,
                            delta, pend_simulate, state_constraints, input_constraints, verbose=2, max_iter=max_iter)

uc = result['uc']
#
if len(state_constraints) > 0:
    x_mpc = sim(zt, np.hstack([ut, uc]), w_mpc, theta_mpc)
    hx = np.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = np.mean(hx > 0, axis=1)
    print('State constraint satisfaction')
    print(cx)
if len(input_constraints) > 0:
    cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints],axis=1)
    print('Input constraint satisfaction')
    print(cu >= 0)
#
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(x_mpc[1,:, i*4], label='MC forward sim')
    if i==1:
        plt.title('MPC solution over horizon')
    # plt.axvline(x_star, linestyle='--', color='g', linewidth=2, label='target')
    # plt.axvline(x_ub, linestyle='--', color='r', linewidth=2, label='upper bound')
    plt.xlabel('t+'+str(i*4+1))
plt.tight_layout()
plt.legend()
plt.show()

plt.plot(uc[0,:-1])
plt.title('MPC determined control action')
plt.show()


plt.plot(x_mpc[0,:,:].mean(axis=0))
plt.show()

plt.plot(x_mpc[1,:,:].mean(axis=0))
plt.show()

plt.plot(x_mpc[0,:,:].mean(axis=0))
plt.show()
#
#
#
#
#
#
