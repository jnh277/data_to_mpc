import numpy as np
# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config

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
    cos_xt1 = jnp.cos(xt[1]) # there are 5 of these
    sin_xt1 = jnp.sin(xt[1]) # there are 4 of these
    m11 = t['Jr + Mp * Lr * Lr'] + t['0.25 * Mp * Lp * Lp'] - t['0.25 * Mp * Lp * Lp'] * cos_xt1 * cos_xt1
    m12 = t['0.5 * Mp * Lp * Lr'] * cos_xt1
    m22 = t['m22'] # this should be a scalar anyway - can be vector
    sc = m11 * m22 - m12 * m12
    tau = (t['Km'] * (u[0] - t['Km'] * xt[2])) / t['Rm'] # u is a scalr
    d1 = tau - t['Dr'] * xt[2] - 2 * t['0.25 * Mp * Lp * Lp'] * sin_xt1 * cos_xt1 * xt[2] * xt[3] + t['0.5 * Mp * Lp * Lr'] * sin_xt1 * xt[3] * xt[3]
    d2 = -t['Dp'] * xt[3] + t['0.25 * Mp * Lp * Lp'] * cos_xt1 * sin_xt1 * xt[2] * xt[2] - t['0.5 * Mp * Lp * g'] * sin_xt1
    dx = jnp.zeros_like(xt)
    dx = index_update(dx, index[0], xt[2])
    dx = index_update(dx, index[1], xt[3])
    dx = index_update(dx, index[2], (m22 * d1 - m12 * d2)/sc)
    dx = index_update(dx, index[3], (m11 * d2 - m12 * d1)/sc)
    return dx
    
def rk4(xt,ut,theta):
    h = theta['h']
    k1 = qube_gradient(xt,ut,theta)
    k2 = qube_gradient(xt + k1*h/2,ut,theta)
    k3 = qube_gradient(xt + k2*h/2,ut,theta)
    k4 = qube_gradient(xt + k3*h,ut,theta)
    temp = xt + (k1/6 + k2/3 + k3/3 + k4/6)*h
    return temp.flatten() # should handle a 2D x just fine

def solve_dare(Q,R,A0,B): # http://dx.doi.org/10.1080/00207170410001714988
    eps = 10 ** -5
    Nx = A0.shape[0]
    Nu = B.shape[1]
    H = Q
    A = A0
    Rinv = np.linalg.inv(R)
    G = B @ Rinv @ B.T
    I = np.identity(Nx)
    while True:
        S = I - G @ H
        S = np.linalg.inv(S)
        Ap = A @ S @ A
        Gp = G + A @ S @ G @ A.T
        Hp = H + A.T @ H @ S @ A
        top = Hp - H
        rat = np.linalg.det(top)/np.linalg.det(Hp)
        if rat <= eps:
            break
    return Hp

# Control parameters
z_star = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 1             # number of samples we will use for MC MPC
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

theta_true = fill_theta(theta_true)

xt_bar = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float).flatten()
ut_bar = np.array([[0]],dtype=float).flatten()

xiso = lambda xt: rk4(xt,ut_bar,theta_true)
uiso = lambda ut: rk4(xt_bar,ut,theta_true)

A = jacfwd(rk4,argnums=0)(xt_bar,ut_bar,theta_true)
print(A)
B = jacfwd(rk4,argnums=1)(xt_bar,ut_bar,theta_true)
print(B)
Q = sqc @ sqc.T
R = src @ src.T
P = solve_dare(Q,R,A,B)
print(P)

