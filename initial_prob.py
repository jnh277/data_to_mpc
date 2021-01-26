import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace
from pathlib import Path
import pickle
from scipy.optimize import minimize

# aim of this script is to solve a single step of the MPC problem based on estimates from
# 100 steps of measurements

# how would we now choose to do control
x_star = 1.0        # desired set point
M = 200     # number of samples we will use for MC MPC
N = 20      # horizonline of MPC algorithm
qc = 1.0    # cost on state error
rc = 1.    # cost on control action

# simulation model
def ssm(x, u, a=0.9, b=0.1):
    return a*x + b*u

# simulation parameters
T = 150             # total simulation time
T_init = 100         # initial number of time steps to record measurements for
x0 = 3.0        # initial x
r_true = 0.1         # measurement noise standard deviation
q_true = 0.05         # process noise standard deviation


# simulate the system
x = np.zeros(T)
x[0] = x0
w = np.random.normal(0.0, q_true, T_init)        # make a point of predrawing noise

# create some inputs that are random but held for 10 time steps
u = np.random.uniform(-0.5,0.5, T)
u = np.reshape(u, (-1,10))
u[:,1:] = np.atleast_2d(u[:,0]).T * np.ones((1,9))
u = u.flatten()

for k in range(T_init):
    x[k+1] = ssm(x[k], u[k]) + w[k]

# simulate measurements
y = np.zeros(T)
y[:T_init] = x[:T_init] + np.random.normal(0.0, r_true, T_init)

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
    'N':T_init,
    'y':y[:T_init],
    'u':u[:T_init],
}



fit = model.sampling(data=stan_data, warmup=1000, iter=2000)
traces = fit.extract()

# settings for 'online' hmc
warmup = 1000
chains = 4
iter = warmup + int(M / chains)

# state samples
z = traces['z']

# parameter samples
a = traces['a']
b = traces['b']
r = traces['r']
q = traces['q']

plot_trace(a,4,1,'a')
plt.title('Initial parameter estimates')
plot_trace(b,4,2,'b')
plot_trace(r,4,3,'r')
plot_trace(q,4,4,'q')
plt.show()


def col_vec(x):
    return np.reshape(x, (-1,1))

def row_vec(x):
    return np.reshape(x, (1, -1))

def simulate(xt, u, a, b, w):
    N = len(u)
    M = len(a)
    x = np.zeros((M, N+1))
    x[:, 0] = xt
    for k in range(N):
        x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
    return x[:, 1:]

def expectation_cost(uc, ut, xt, x_star, a, b, w, qc, rc):
    u = np.hstack([ut, uc]) # u_t was already performed, so u is N-1
    x = simulate(xt, u, a, b, w)
    V = np.sum((qc*(x - x_star)) ** 2) + np.sum((rc * uc)**2)
    return V

# input bounds
bnds = ((-3.0, 3.0),)*(N-1)

uc = np.zeros(N-1)  # initialise uc

for t in range(T_init-1,T-1):
    # simulate system
    # apply u_t
    # this takes x_t to x_{t+1}
    # measure y_t
    x[t+1] = ssm(x[t], u[t]) + q_true * np.random.randn()
    y[t] = x[t] + r_true * np.random.randn()

    # estimate system (estimates up to x_t)
    stan_data = {
        'N': t+1,
        'y': y[:t+1],
        'u': u[:t+1],
    }
    fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains)
    traces = fit.extract()

    # state samples
    z = traces['z']

    # parameter samples
    a = traces['a']
    b = traces['b']
    r = traces['r']
    q = traces['q']

    # we also need to sample noise
    w = np.reshape(q, (-1, 1)) * np.random.randn(M, N)  # uses the sampled stds

    # determine next control action u_{t+1}
    xt = z[:, t]
    ut = u[t]
    cost = lambda uc: expectation_cost(uc, ut, xt, x_star, a, b, w, qc, rc)
    uc = np.hstack([uc[1:],0.0])
    res = minimize(cost, uc, bounds=bnds)
    uc = res.x
    u[t+1] = uc[0]

plt.subplot(2,1,1)
plt.plot(u)
plt.axvline(100, linestyle='--', color='k', linewidth=2)

plt.subplot(2, 1, 2)
plt.plot(x)
plt.axhline(x_star, linestyle='--', color='r', linewidth=2)
plt.axvline(100, linestyle='--', color='k', linewidth=2)

plt.show()










