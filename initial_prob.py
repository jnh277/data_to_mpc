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

Current set up: Uses MC to give an expected cost and then satisfies the constraints with all
particles

Implementation:
This implementation uses scipy's optimisation routines with finite differences for gradients
"""

import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, row_vec, col_vec
from pathlib import Path
import pickle
from scipy.optimize import minimize, NonlinearConstraint

# Control parameters
x_star = 1.0        # desired set point
M = 200             # number of samples we will use for MC MPC
N = 20              # horizonline of MPC algorithm
qc = 1.0            # cost on state error
rc = 1.             # cost on control action
x_ub = 1.05         # upper bound constraint on state

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
x0 = 3.0            # initial x
r_true = 0.1        # measurement noise standard deviation
q_true = 0.05       # process noise standard deviation

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.9, b=0.1):
    return a*x + b*u

x = np.zeros(T+1)
x[0] = x0                                   # initial state
w = np.random.normal(0.0, q_true, T)        # make a point of predrawing noise

# create some inputs that are random but held for 10 time steps
u = np.random.uniform(-0.5,0.5, T)
u = np.reshape(u, (-1,10))
u[:,1:] = np.atleast_2d(u[:,0]).T * np.ones((1,9))
u = u.flatten()

for k in range(T):
    x[k+1] = ssm(x[k], u[k]) + w[k]

# simulate measurements
y = np.zeros(T)
y = x[:T] + np.random.normal(0.0, r_true, T)

plt.subplot(2,1,1)
plt.plot(u)
plt.title('Simulated inputs used for inference')

plt.subplot(2, 1, 2)
plt.plot(x)
plt.plot(y,linestyle='None',color='r',marker='*')
plt.title('Simulated state and measurements used for inferences')
plt.tight_layout()
plt.show()

#----------- USE HMC TO PERFORM INFERENCE ---------------------------#
# avoid recompiling
model_name = 'LSSM_demo'
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
}

fit = model.sampling(data=stan_data, warmup=1000, iter=2000)
traces = fit.extract()

# state samples
z = traces['z']

# parameter samples
a = traces['a']
b = traces['b']
r = traces['r']
q = traces['q']

# plot the initial parameter marginal estimates
plot_trace(a,4,1,'a')
plt.title('HMC inferred parameters')
plot_trace(b,4,2,'b')
plot_trace(r,4,3,'r')
plot_trace(q,4,4,'q')
plt.show()

# plot some of the initial marginal state estimates
for i in range(4):
    if i==1:
        plt.title('HMC inferred states')
    plt.subplot(2,2,i+1)
    plt.hist(z[:, i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
    plt.axvline(x[i*20+1], label='True', linestyle='--',color='k',linewidth=2)
    plt.xlabel('x_'+str(i+1))
plt.tight_layout()
plt.legend()
plt.show()

# ----- Solve the MC MPC control problem ------------------#

# function to forward simulate a particle / scenario
def simulate(xt, u, a, b, w):
    N = len(u)
    M = len(a)
    x = np.zeros((M, N+1))
    x[:, 0] = xt
    for k in range(N):
        x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
    return x[:, 1:]

# function to calculate the expectation cost
def expectation_cost(uc, ut, xt, x_star, a, b, w, qc, rc):
    u = np.hstack([ut, uc]) # u_t was already performed, so u is N-1
    x = simulate(xt, u, a, b, w)
    V = np.sum((qc*(x - x_star)) ** 2) + np.sum((rc * uc)**2)
    return V

# input bounds
bnds = ((-3.0, 3.0),)*(N-1)
# initialise uc
uc = np.zeros(N-1)


# downsample the the HMC output since for illustration purposes we sampled > M
ind = np.random.choice(len(a), M, replace=False)
a = a[ind]      # same indices for all to ensure they correpond to the same realisation from dist
b = b[ind]
q = q[ind]
r = r[ind]

xt = z[ind,-1]       # inferred state for current time step

# we also need to sample noise
w = col_vec(q) * np.random.randn(M, N)  # uses the sampled stds

ut = u[-1]      # the last control action taht was applied and will take x_t to x_{t+1}
cost = lambda uc: expectation_cost(uc, ut, xt, x_star, a, b, w, qc, rc)

# add an output constraint to x_{t+2} (x_{t+1} is already decided)
con = lambda u: (simulate(xt, np.hstack([ut, u]), a, b, w)[:, 1:]).flatten()
lb = -5 * np.ones((M * (N - 1)))
ub = x_ub * np.ones((M * (N - 1)))
nlc = NonlinearConstraint(con, lb, ub)

res = minimize(cost, uc, bounds=bnds, constraints=(nlc))
print(res)
uc = res.x

# recreate the forward simulation that the MPC controller has converged to with its choice of
# uc
x_mpc = simulate(xt, np.hstack([ut, uc]), a, b, w)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(x_mpc[:, i*3], label='MC forward sim')
    if i==1:
        plt.title('MPC solution over horizon')
    plt.axvline(1.0, linestyle='--', color='g', linewidth=2, label='target')
    plt.axvline(x_ub, linestyle='--', color='r', linewidth=2, label='constraint')
    plt.xlabel('t+'+str(i*3+1))
plt.tight_layout()
plt.legend()
plt.show()

plt.plot(uc)
plt.title('MPC determined control action')
plt.show()




## potential framework for doing this in a loop
# for t in range(T_init-1,T-1):
    # simulate system
    # apply u_t
    # this takes x_t to x_{t+1}
    # measure y_t
    # x[t+1] = ssm(x[t], u[t]) + q_true * np.random.randn()
    # y[t] = x[t] + r_true * np.random.randn()

    # estimate system (estimates up to x_t)
    # stan_data = {
    #     'N': t+1,
    #     'y': y[:t+1],
    #     'u': u[:t+1],
    # }
    # fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains)
    # traces = fit.extract()

    # state samples
    # z = traces['z']
    #
    # # parameter samples
    # a = traces['a']
    # b = traces['b']
    # r = traces['r']
    # q = traces['q']

    # we also need to sample noise
    # w = np.reshape(q, (-1, 1)) * np.random.randn(M, N)  # uses the sampled stds
    #
    # # determine next control action u_{t+1}
    # xt = z[:, t]
    # ut = u[t]
    # cost = lambda uc: expectation_cost(uc, ut, xt, x_star, a, b, w, qc, rc)
    # uc = np.hstack([uc[1:],0.0])
    # res = minimize(cost, uc, bounds=bnds)
    # uc = res.x
    # u[t+1] = uc[0]











