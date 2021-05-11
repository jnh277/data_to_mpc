"""
Simulates a mass-spring-damper system in one dimension (2 state system) with random noise on the state transition. 
Generates measurements of the system corresponding to position and acceleration of the mass, with random noise on each measurement.
At time t, the system model changes from system1 --> system 2, in the simplest case this is a change in the parameters.
Given a window of data where the switch occurs at time t, pystan will sample the states, parameters for both systems, and switching time t jointly.
"""
import os
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")
# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace_grid
from pathlib import Path
import pickle

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.lax import scan
from jax.ops import index, index_update
from jax.config import config
config.update("jax_enable_x64", True)

if __name__ == "__main__":
    plot_bool = True
    #----------------- Parameters ---------------------------------------------------#

    T = 200             # number of time steps to simulate and record measurements for
    Ts = 0.004
    cms = 100
    # true (simulation) parameters
    z1_0 = cms*0.009  # initial position
    z2_0 = cms*0.0  # initial velocity
    # r1_true = cms*0.0000002 # measurement noise standard deviation
    r1_true = 0.005 # cm stdev
    # q1_true = cms*0.0005 * Ts # process noise standard deviation
    # q2_true = cms*0.00005 * Ts
    q1_true = 0.1 * Ts
    q2_true = 0.01 * Ts
    # q2_true = 0.1

    # got these values from the data sheet
    Mb_true = 0.06 # kg, mass of the steel ball
    Ldiff_true = 0.04 * cms * cms # 100*100*H, or kg * cm^2 ­* s^-2 *­ A^-2 difference between coil L at zero and infinity (L(0) - L(inf))
    x50_true = cms * 0.002 # cm, location in mode of L where L is 50% of the infinity and 0 L's
    grav = 9.81 * cms # gravity in cm/s/s
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

    #----------------- Simulate the system-------------------------------------------#

    def maglev_gradient(xt, u, t): # uses the lumped parametersation 
        dx = jnp.zeros_like(xt)
        dx = index_update(dx, index[0, :], xt[1, :])
        dx = index_update(dx, index[1, :], t['g'] - t['k0'] * u * u / (t['I0'] + xt[0,:]) / (t['I0'] + xt[0,:]))
        return dx

    def current_current(xt,t): # expect a slice of x_hist to be xt, shape (2,)
        return np.sqrt(t['g']/t['k0'])*(xt[0] + t['I0'])

    def rk4(xt, ut, theta):
        h = theta['h']
        k1 = maglev_gradient(xt, ut, theta)
        k2 = maglev_gradient(xt + k1 * h / 2, ut, theta)
        k3 = maglev_gradient(xt + k2 * h / 2, ut, theta)
        k4 = maglev_gradient(xt + k3 * h, ut, theta)
        return xt + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * h  # should handle a 2D x just fine

    @jit
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


    z_sim = np.zeros((Nx,1,T+1), dtype=float) # state history allocation

    # load initial state
    z_sim[0,0,0] = z1_0 
    z_sim[1,0,0] = z2_0 

    # noise is predrawn and independant
    w_sim = np.zeros((Nx,1,T),dtype=float)
    w_sim[0,0,:] = np.random.normal(0.0, q1_true, T)
    w_sim[1,0,:] = np.random.normal(0.0, q2_true, T)

    # draw measurement noise
    v = np.zeros((Ny,T+1), dtype=float)
    # v[0,:] = np.random.normal(0.0, r1_true, T+1)
    v[0,:] = np.random.standard_t(5, T + 1) * r1_true

    # simulated measurements 
    y = np.zeros((Ny,T+1), dtype=float)

    # some random error state PID that stabilises the system
    Kp = 10 / cms 
    Ki = 2 / cms 
    Kd = 2 / cms 
    reference = z1_0 - cms*0.002
    u = np.zeros((Nu,T), dtype=float)
    y[0,0] = z1_0 + v[0,0]
    error = y[0,0] - reference
    integrator = 0
    difference = 0
    u[:,0] = current_current(y[[0],0],theta_true) + Kp * error +  Ki * integrator + Kd * difference
    for k in np.arange(T):
        z_sim[:,:,[k+1]] = simulate(z_sim[:,:,k],u[:,[k]],w_sim[:,:,[k]],theta_true)
        y[0,k+1] = z_sim[0,0,k+1] + v[0,k+1]
        if k < T - 1:
            difference = -error
            error = y[0,k+1] - reference
            integrator += error * Ts
            difference += error
            difference /= Ts
            if k > T/2:
                reference = z1_0
                error = y[0,k+1] - reference
            u[:,k+1] = current_current(y[[0],k+1],theta_true) + Kp * error +  Ki * integrator + Kd * difference

    

    y = y[[0],:-1]
    # y[0,:] = z_sim[0,0,:-1]
    # y = y + v; # add noise to measurements

    plt.subplot(2,1,1)
    plt.plot(u[0,:])
    plt.title('Simulated inputs and measurement used for inference')

    plt.subplot(2, 1, 2)
    plt.plot(z_sim[0,0,:])
    plt.plot(y[0,:],linestyle='None',color='r',marker='*')
    plt.title('Simulated state 1 and measurements used for inferences')
    plt.tight_layout()
    plt.show()


    #----------- USE HMC TO PERFORM INFERENCE ---------------------------#
    # avoid recompiling
    script_path = os.path.dirname(os.path.realpath(__file__))
    model_name = 'maglev2'
    path = '/stan/'
    if Path(script_path+path+model_name+'.pkl').is_file():
        model = pickle.load(open(script_path+path+model_name+'.pkl', 'rb'))
    else:
        model = pystan.StanModel(file=script_path+path+model_name+'.stan')
        with open(script_path+path+model_name+'.pkl', 'wb') as file:
            pickle.dump(model, file)

    stan_data = {
        'no_obs': T,
        'N':T,
        'y':y[0,:],
        'u':u[0,:],
        'g':grav,
        'z0_mu': np.array([z1_0,z2_0]),
        'z0_std': np.array([0.2,0.2]),
        'theta_p_mu': np.array([I0_true, k0_true]),
        'theta_p_std':1.0*np.array([I0_true, k0_true]),
        'r_p_mu': np.array([r1_true]),
        'r_p_std': np.array([0.1]),
        'q_p_mu': np.array([q1_true, q2_true]),
        'q_p_std': np.array([0.1,0.1]),
        'Ts':Ts
    }

    v_init = np.zeros((1, T + 1))
    span = 13  # has to be odd
    for tt in range(T):
        if tt - (span // 2) < 0:
            ind_start = 0
            ind_end = span
        elif tt + (span // 2) + 1 > T:
            ind_end = T
            ind_start = T - span - 1
        else:
            ind_start = tt - (span // 2)
            ind_end = tt + (span // 2) + 1
        p = np.polyfit(np.arange(ind_start, ind_end), y[0, np.arange(ind_start, ind_end)], 2)
        # v = np.polyval(p,np.arange(ind_start,ind_end))
        # plt.plot(v)
        # plt.plot(y[0,ind_start:ind_end])
        # plt.show()
        v_init[0, tt] = (2 * p[0] * tt + p[1]) / Ts

    v_init[0, -1] = v_init[0, -2]

    h_init = np.zeros((2, T+1))
    h_init[0, :-1] = y[0, :]
    h_init[0, -1] = y[0,-1]
    h_init[1, :] = v_init[0,:]     # smoothed gradients of measurements
    theta_init = np.array([I0_true, k0_true]) # start theta somewhere?



    # plt.plot(v_init[0,:])
    # plt.plot(z_sim[1,0,:])
    # plt.show()

    def init_function(ind):
        output = dict(theta=theta_init,
                      h=h_init,
                      # q=last_pos[ind]['q'],
                      # r=last_pos[ind]['r']
                      )
        return output

    init = [init_function(0),init_function(1),init_function(2),init_function(3)]

    fit = model.sampling(data=stan_data, warmup=2000, iter=3000, chains=4, init=init)
    traces = fit.extract()

    # state samples
    z_samps = np.transpose(traces['h'],(1,0,2)) # Ns, Nx, T --> Nx, Ns, T
    theta_samps = traces['theta']

    # parameter samples
    I0_samps = theta_samps[:, 0].squeeze()
    k0_samps = theta_samps[:, 1].squeeze()
    r_samps = traces['r'].transpose()
    q_samps = traces['q'].transpose()

    # plot the initial parameter marginal estimates
    q1plt = q_samps[0,:].squeeze()
    q2plt = q_samps[1,:].squeeze()
    r1plt = r_samps[0,:].squeeze()

    plot_trace_grid(I0_samps,1,5,1,'I0',true_val=I0_true)
    plt.title('HMC inferred parameters')
    plot_trace_grid(k0_samps,1,5,2,'k0',true_val=k0_true)
    plot_trace_grid(q1plt,1,5,3,'q1',true_val=q1_true)
    plot_trace_grid(q1plt,1,5,4,'q2',true_val=q2_true)
    plot_trace_grid(r1plt,1,5,5,'r1',true_val=r1_true)
    plt.show()

    # # plot some of the initial marginal state estimates
    # for i in range(4):
    #     if i==1:
    #         plt.title('HMC inferred position')
    #     plt.subplot(2,2,i+1)
    #     plt.hist(z_samps[0,:,i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
    #     plt.axvline(z_sim[0,i*20+1], label='True', linestyle='--',color='k',linewidth=2)
    #     plt.xlabel('x_'+str(i+1))
    # plt.tight_layout()
    # plt.legend()
    # plt.show()


    plt.plot(z_samps[0,:,:].mean(axis=0))
    plt.plot(z_sim[0,0,:])
    plt.show()



