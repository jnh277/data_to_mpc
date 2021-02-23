
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.scipy.special import expit
from dare import dare_P

from helpers import row_vec


def logbarrier(cu, mu):       # log barrier for the constraint c(u,s,epsilon) >= 0
    return jnp.sum(-mu * jnp.log(cu))

def chance_constraint(hu, epsilon, gamma):    # Pr(h(u) >= 0 ) >= (1-epsilon)
    return jnp.mean(expit(hu / gamma), axis=1) - (1 - epsilon)     # take the sum over the samples (M)

def log_barrier_cost(z, ut, xt, x_star, theta, w, sqc, src, delta, mu, gamma, simulate, state_constraints, input_constraints, N, Nu):
    """
    log barrier cost function for solving the MPC optimisation problem with state chance constraints
    and input constraints.

    Definitions:
        o: number of states
        N: horizon we wish to optimise the control inputs over
        M: number of samples in the Monte Carlo approximation
        ncx: number of state constraints (each state constraint is applied over the entire horizon)
        ncu: number of input constraints (each input constraint is applied over the entire horizon)
        Nu: dimension of the input
    Inputs
        z: array of parameters to be optimised, contains the control inputs (uc), and the chance constraint
            slack variables (epsilon). uc is size [Nu,N], and epsilon is size [ncx*N,]. These are flattened
            and stacked.
        ut: control action applied at current time t, that takes current state x_t to next state x_t
            has size [Nu,1] or [Nu,]
        xt: current state
        x_star: desired state - array of size [o,] or [o,1] to apply the same target over entire horizon
                OR can be size [o, N] to apply a different target at each time step over horizon
        theta: dictionary of model parameters, should be M samples of each parameter
                (stacked in array or some such)
        w: process noise - array of size [o,M,N+1]
        sqc: square root of the state error weighting matrix - size [o,o]
        src: square root of the input weighting matrix - size [1,1]
        delta: target maximum probability of state constraint violation - scalar or array
                of size [ncx * N,]. Scalar specifies one value for all constraints over entire horizon
                Array specifies individual values for each constraint at timestep point in horizon
        mu: log barrier parameter - scalar
        gamma: sigmoid indicator function approximation parameter - scalar
        simulate: function to simulate the evolution of the systems states
            matching template
            def simulate(xt, u, w, theta)
                return array of size [o,M,N] containing x_{t+1},...,x_{t+1+N}
            where u = [ut,uc]
            Must contain only jax compatible functions.
        state_constraints: tuple or list of state constraint functions h_i(x), which
            the optimisation will satisfy h_i(x) > 0 with probability 1-delta.
            Each h_i(x) is applied to the entire horizon and needs to return an array
            of size [o,M,N]. Must contain only jax compatible functions.
        input_constraints: tuple or list of input constraint functions c_i(u), which
            the optimisation will make satisfy c_i(u) > 0. Each c_i(u) is applied to
            the entire horizon and needs to return an array of size [Nu,M,N].
            Must contain only jax compatible functions.
        N: the horizon (must be input to avoid dynamic sizing of arrays)
        Nu: dimension of the input

    returns the cost

    """


    # print('Compiling log barrier cost')
    ncu = len(input_constraints)    # number of input constraints
    ncx = len(state_constraints)    # number of state constraints

    o = w.shape[0]
    uc = jnp.reshape(z[:Nu*N], (Nu, N))              # control input variables  #,
    epsilon = z[Nu*N:N*Nu+ncx*N]                        # slack variables on state constraints

    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    # state error and input penalty cost and cost that drives slack variables down
    # potentially np.reshape(x_star,(o,1,-1))??
    V1 = jnp.sum(jnp.matmul(sqc,jnp.reshape(x[:,:,1:] - jnp.reshape(x_star,(o,1,-1)),(o,-1))) ** 2) + jnp.sum(jnp.matmul(src, uc)**2) + jnp.sum(300 * (epsilon + 1e3)**2)
    # need a log barrier on each of the slack variables to ensure they are positve
    V2 = logbarrier(epsilon - delta, mu)     # aiming for 1-delta% accuracy
    # now the chance constraints
    if ncx > 0:
        hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
        cx = chance_constraint(hx, epsilon, gamma)
    else:
        cx = 1.0    # ln(cx,mu) = 0

    if ncu > 0:
        cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
    else:
        cu = 1.0        # ln(cu,mu) = 0
    V3 = logbarrier(cx, mu) + logbarrier(cu, mu)
    ferr = jnp.reshape(x[:,:,[-1]] - jnp.reshape(x_star,(o,1,-1)),(o,-1)).squeeze()
    V4 = jnp.cumsum(ferr.T @ theta['P'] @ ferr).flatten()
    return V1 + V2 + V3 + V4.squeeze()

def solve_chance_logbarrier(uc0, cost, gradient, hessian, ut, xt, theta, w, x_star, sqc, src, delta, simulate,
                            state_constraints, input_constraints, verbose=True, max_iter=1000, gamma=1.0, mu=1e4,
                            epsilon0=1.0, c1=1e-4):
    """
    Solves the MPC problem with chance state constraints and regular input constraints given
    a log barrier formulation of the cost function

    Definitions:
        o: number of states
        N: horizon we wish to optimise the control inputs over
        M: number of samples in the Monte Carlo approximation
        ncx: number of state constraints (each state constraint is applied over the entire horizon)
        ncu: number of input constraints (each input constraint is applied over the entire horizon)
        Nu: dimension of the input
    Inputs
        uc0: Starting values for the optimisation of the inputs (must be feasible) size [1,N]
        cost: compiled version of log_barrier_cost
        gradient: compiled gradient of log_barrier_cost function
        hessian: compiled hessian of log_barrier_cost function
        ut: control action applied at current time t, that takes current state x_t to next state x_t
            has size [Nu,1] or [Nu,]
        xt: current state
        x_star: desired state - array of size [o,] or [o,1]
        theta: dictionary of model parameters
        w: process noise - array of size [o,M,N+1]
        sqc: square root of the state error weighting matrix - size [o,o]
        src: square root of the input weighting matrix - size [Nu,Nu]
        delta: target maximum probability of state constraint violation - scalar
        mu: log barrier parameter - scalar
        gamma: sigmoid indicator function approximation parameter - scalar
        simulate: function to simulate the evolution of the systems states
            matching template
            def simulate(xt, u, w, theta)
                return array of size [o,M,N] containing x_{t+1},...,x_{t+1+N}
            where u = [ut,uc]
            Must contain only jax compatible functions.
        state_constraints: tuple or list of state constraint functions h_i(x), which
            the optimisation will satisfy h_i(x) > 0 with probability 1-delta.
            Each h_i(x) is applied to the entire horizon and needs to return an array
            of size [o,M,N]. Must contain only jax compatible functions.
        input_constraints: tuple or list of input constraint functions c_i(u), which
            the optimisation will make satisfy c_i(u) > 0. Each c_i(u) is applied to
            the entire horizon and needs to return an array of size [Nu,M,N].
            Must contain only jax compatible functions.
        verbose [optional, default=True]: If true, displays start and exit condition,
            if verbose==2, displays iteration information
            if verbose==0 or False, runs silently
        max_iter [optional, default=1e4]: maximum number of iterations
        mu [optional, default=1e3]: starting value for parameter of log barrier function
        gamma [optional, default=1.0]: starting value for parameter of sigmoid function
        epsilon0 [optional, default=1.0]: starting value for chance state constraint slack
            variables
        c1 [optional, default=1e-4]: first wolfe condition parameter

    returns result dict containing:
        'uc': optimised input actions
        'epsilon': the chance state constraint slack variables
        'status': exit condition
            status==0: termination criteria satisfied and all constraints satisfied
            status=1: termination criteria satisfied but desired prbability of
                    state constraint not achieved
            status=2: termination criteria satisfied but input constraints violated
            status=3: termination criteria not satisfied could not find an alpha value
                    to give a valid line search result.
        'gamma':final value of the gamma parameter,
        'mu':final value of the mu parameter,
        'gradient': gradient of the log barrier cost function on completion,
        'hessian': hessian of the log barrier cost function on completion,
        'newton_decrement': calculated as np.dot(g,p)

    """
    if verbose:
        print('Starting optimisation')

    o,N = w.shape[0],w.shape[2]-1
    Nu = uc0.shape[0]             # input dimension
    ncu = len(input_constraints)  # number of input constraints
    ncx = len(state_constraints)  # number of state constraints

    z = np.hstack([uc0.flatten(), epsilon0*np.ones((ncx * N,))])

    args = (device_put(ut), device_put(xt), device_put(x_star), device_put(theta),
            device_put(w), device_put(sqc), device_put(src), device_put(delta))

    jmu = device_put(mu)
    jgamma = device_put(gamma)
    status=7

    for i in range(max_iter):
        # compute cost, gradient, and hessian
        jz = device_put(z)
        c = np.array(cost(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N, Nu))
        g = np.array(gradient(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N, Nu))
        h = np.array(hessian(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N, Nu))

        # compute search direction
        try:
            p = - np.linalg.solve(h, g)
            # calculate newton decrement
            nd = np.dot(p,g)
            # check that we have a valid search direction and if not then fix
            if nd >= -1e-8:
                [d, v] = np.linalg.eig(h)
                if np.iscomplex(d).any():       # stop complex eigenvalues
                    d = np.real(d)
                    v = np.real(v)
                ind = d < 1e-5
                d[ind] = 1e-5 + np.abs(d[ind])
                hn = v @ np.diag(d) @ v.T
                p = - np.linalg.solve(hn, g)
                nd = np.dot(p,g)
                if jnp.iscomplex(nd):
                    status=6
                    print('search direction became complex valued')
                    break
        except:
            status=8
            print('linalg.solve error')
            break

        # perform line search
        alpha = 1.0
        for k in range(52):
            ztest = z + alpha * p
            ctest = np.array(cost(device_put(ztest), *args, jmu, jgamma, simulate, state_constraints, input_constraints, N, Nu))
            # if np.isnan(ctest) or np.isinf(ctest): nan and inf checks should be redundant
            if ctest < (c + c1 * alpha * nd):  # first wolfe condition
                z = ztest
                break

            alpha = alpha / 2

        if k == 51 and np.abs(nd) > 1e-2:
            status = 3
            print('Unable to find an alpha that decreases cost')
            break
        if verbose==2:
            print('Iter:', i + 1, 'Cost: ', c, 'nd:', np.dot(g, p), 'alpha: ', alpha, 'mu: ', mu, 'gamma: ', gamma)

        if np.abs(nd) < 1e-2:  # if search direction was really small, then decrease mu and s for next iteration
            if mu < 1e-6 and gamma < 1e-3:
                status = 0
                break  # termination criteria satisfied

            mu = max(mu / 2, 0.999e-6)
            gamma = max(gamma / 1.25, 0.999e-3)
            # need to adjust the slack after changing gamma
            if ncx > 0:
                x_new = simulate(xt, np.hstack([ut, row_vec(z[:Nu*N])]), w, theta)
                hx = jnp.concatenate([state_constraint(x_new[:, :, 1:]) for state_constraint in state_constraints], axis=2)
                cx = chance_constraint(hx, z[Nu*N:Nu*N + ncx * N], gamma)
                z[Nu*N:Nu*N + ncx * N] += -np.minimum(cx[0, :], 0) + 1e-6
            jmu = device_put(mu)
            jgamma = device_put(gamma)


    uc = jnp.reshape(z[:Nu*N], (Nu, N))                 # control input variables  #,
    epsilon = z[Nu*N:N*Nu+ncx*N]                        # slack variables on state constraints

    if status==0:
        if verbose:
            print("Converged to minima")
        x_check = simulate(xt, np.hstack([ut, uc]), w, theta)
        if ncx > 0:
            hx = jnp.concatenate([state_constraint(x_check[:, :, 1:]) for state_constraint in state_constraints], axis=2)
            cx = np.mean(hx > 0, axis=1)
            if (cx < (1-delta-1e-4)).any():
                if verbose:
                    print("State constraints not satisfied with desired probability")
                    print("Lowest state constraint satisfaction probability of ", cx.min())
                status=1
        if ncu > 0:
            cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
            if (cu < 1e-8).any():
                status=2
                if verbose:
                    print("Input constraints violated")

    result = {'uc':uc,
              'epsilon':epsilon,
              'status':status,
              'gamma':gamma,
              'mu':mu,
              'gradient':g,
              'hessian':h,
              'newton_decrement':np.dot(g,p)}
    return result



def log_barrier_cosine_cost(z, ut, xt, x_star, theta, w, sqc, src, delta, mu, gamma, simulate, state_constraints, input_constraints, N, Nu):
    """
    log barrier cost function for solving the MPC optimisation problem with state chance constraints
    and input constraints.

    Definitions:
        o: number of states
        N: horizon we wish to optimise the control inputs over
        M: number of samples in the Monte Carlo approximation
        ncx: number of state constraints (each state constraint is applied over the entire horizon)
        ncu: number of input constraints (each input constraint is applied over the entire horizon)
        Nu: dimension of the input
    Inputs
        z: array of parameters to be optimised, contains the control inputs (uc), and the chance constraint
            slack variables (epsilon). uc is size [Nu,N], and epsilon is size [ncx*N,]. These are flattened
            and stacked.
        ut: control action applied at current time t, that takes current state x_t to next state x_t
            has size [Nu,1] or [Nu,]
        xt: current state
        x_star: desired state - array of size [o,] or [o,1] to apply the same target over entire horizon
                OR can be size [o, N] to apply a different target at each time step over horizon
        theta: dictionary of model parameters, should be M samples of each parameter
                (stacked in array or some such)
        w: process noise - array of size [o,M,N+1]
        sqc: square root of the state error weighting matrix - size [o,o]
        src: square root of the input weighting matrix - size [1,1]
        delta: target maximum probability of state constraint violation - scalar or array
                of size [ncx * N,]. Scalar specifies one value for all constraints over entire horizon
                Array specifies individual values for each constraint at timestep point in horizon
        mu: log barrier parameter - scalar
        gamma: sigmoid indicator function approximation parameter - scalar
        simulate: function to simulate the evolution of the systems states
            matching template
            def simulate(xt, u, w, theta)
                return array of size [o,M,N] containing x_{t+1},...,x_{t+1+N}
            where u = [ut,uc]
            Must contain only jax compatible functions.
        state_constraints: tuple or list of state constraint functions h_i(x), which
            the optimisation will satisfy h_i(x) > 0 with probability 1-delta.
            Each h_i(x) is applied to the entire horizon and needs to return an array
            of size [o,M,N]. Must contain only jax compatible functions.
        input_constraints: tuple or list of input constraint functions c_i(u), which
            the optimisation will make satisfy c_i(u) > 0. Each c_i(u) is applied to
            the entire horizon and needs to return an array of size [Nu,M,N].
            Must contain only jax compatible functions.
        N: the horizon (must be input to avoid dynamic sizing of arrays)
        Nu: dimension of the input

    returns the cost

    """


    # print('Compiling log barrier cost')
    ncu = len(input_constraints)    # number of input constraints
    ncx = len(state_constraints)    # number of state constraints

    o = w.shape[0]
    uc = jnp.reshape(z[:Nu*N], (Nu, N))              # control input variables  #,
    epsilon = z[Nu*N:N*Nu+ncx*N]                        # slack variables on state constraints

    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    # state error and input penalty cost and cost that drives slack variables down
    V1 = jnp.sum((sqc[1,1] * (jnp.cos(x[1,:,1:]) - jnp.cos(x_star[1,0])))**2) + \
         jnp.sum((sqc[0, 0] * (x[0, :, 1:] - x_star[0, 0])) ** 2) + \
         jnp.sum((sqc[2, 2] * (x[2, :, 1:] - x_star[2, 0])) ** 2) + \
         jnp.sum((sqc[3, 3] * (x[3, :, 1:] - x_star[3, 0])) ** 2) + \
         jnp.sum(jnp.matmul(src, uc) ** 2) + jnp.sum(300 * (epsilon + 1e3) ** 2)
    # need a log barrier on each of the slack variables to ensure they are positve
    V2 = logbarrier(epsilon - delta, mu)     # aiming for 1-delta% accuracy
    # now the chance constraints
    if ncx > 0:
        hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
        cx = chance_constraint(hx, epsilon, gamma)
    else:
        cx = 1.0    # ln(cx,mu) = 0

    if ncu > 0:
        cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
    else:
        cu = 1.0        # ln(cu,mu) = 0
    V3 = logbarrier(cx, mu) + logbarrier(cu, mu)
    return V1 + V2 + V3