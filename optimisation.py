
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.scipy.special import expit

from helpers import row_vec


def logbarrier(cu, mu):       # log barrier for the constraint c(u,s,epsilon) >= 0
    return jnp.sum(-mu * jnp.log(cu))

def chance_constraint(hu, epsilon, gamma):    # Pr(h(u) >= 0 ) >= (1-epsilon)
    return jnp.mean(expit(hu / gamma), axis=1) - (1 - epsilon)     # take the sum over the samples (M)

def log_barrier_cost(z, ut, xt, x_star, theta, w, sqc, src, delta, mu, gamma, simulate, state_constraints, input_constraints, N):

    print('Compiling log barrier cost')
    ncu = len(input_constraints)    # number of input constraints
    ncx = len(state_constraints)    # number of state constraints

    o = w.shape[0]
    uc = jnp.reshape(z[:N], (o, -1))                # control input variables  #,
    epsilon = z[N:N+ncx*N]                          # slack variables on state constraint probabilities

    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    # state error and input penalty cost and cost that drives slack variables down
    V1 = jnp.sum(jnp.matmul(sqc,jnp.reshape(x - x_star,(o,-1))) ** 2) + jnp.sum(jnp.matmul(src, uc)**2) + jnp.sum(100 * (epsilon + 1e3)**2)
    # need a log barrier on each of the slack variables to ensure they are positve
    V2 = logbarrier(epsilon - delta, mu)     # aiming for 1-delta% accuracy
    # now the chance constraints
    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma)
    cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
    V3 = logbarrier(cx, mu) + logbarrier(cu, mu)
    return V1 + V2 + V3

def solve_chance_logbarrier(uc0, cost, gradient, hessian, ut, xt, theta, w, x_star, sqc, src, delta, simulate,
                            state_constraints, input_constraints, verbose=True, max_iter=1000, gamma=1.0, mu=1e4,
                            epsilon0=1.0, s0=10.0):

    [o,N] = uc0.shape

    ncu = len(input_constraints)  # number of input constraints
    ncx = len(state_constraints)  # number of state constraints

    z = np.hstack([uc0.flatten(), epsilon0*np.ones((ncx * N,))])

    args = (device_put(ut), device_put(xt), device_put(x_star), device_put(theta),
            device_put(w), device_put(sqc), device_put(src), device_put(delta))

    jmu = device_put(mu)
    jgamma = device_put(gamma)

    for i in range(max_iter):
        # compute cost, gradient, and hessian
        jz = device_put(z)
        c = np.array(cost(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))
        g = np.array(gradient(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))
        h = np.array(hessian(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))

        # compute search direction
        p = - np.linalg.solve(h, g)
        # check that we have a valid search direction and if not then fix
        # TODO: make this less hacky (look at slides)
        beta2 = 1e-8
        while np.dot(p, g) > 0:
            p = - np.linalg.solve((h + beta2 * np.eye(h.shape[0])), g)
            beta2 = beta2 * 2

        # perform line search
        alpha = 1.0
        for k in range(52):
            # todo: (lower priority) need to use the wolfe conditions to ensure a bit of a better decrease
            ztest = z + alpha * p
            ctest = np.array(cost(device_put(ztest), *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))
            # if np.isnan(ctest) or np.isinf(ctest):
            #     continue
            # nan and inf checks should be redundant
            if ctest < c:
                z = ztest
                break

            alpha = alpha / 2

        if k == 51 and np.abs(np.dot(g, p)) > 1e-2:
            status = 3
            print('Unable to find an alpha that decreases cost')
            break
        if verbose==2:
            print('Iter:', i + 1, 'Cost: ', c, 'nd:', np.dot(g, p), 'alpha: ', alpha, 'mu: ', mu, 'gamma: ', gamma)

        if np.abs(
                np.dot(g, p)) < 1e-2:  # if search direction was really small, then decrease mu and s for next iteration
            if mu < 1e-6 and gamma < 1e-3:
                status = 0
                break  # termination criteria satisfied

            mu = max(mu / 2, 0.999e-6)
            gamma = max(gamma / 1.25, 0.999e-3)
            # need to adjust the slack after changing gamma
            # x_new = simulate(xt, np.hstack([ut, row_vec(z[:N - 1])]), w, theta)
            x_new = simulate(xt, np.hstack([ut, row_vec(z[:N])]), w, theta)
            hx = jnp.concatenate([state_constraint(x_new[:, :, 1:]) for state_constraint in state_constraints], axis=2)
            # cx = chance_constraint(hx, z[N - 1:(N - 1) + ncx * (N - 1)], gamma)
            cx = chance_constraint(hx, z[N:N + ncx * N], gamma)
            z[N:N + ncx * N] += -np.minimum(cx[0, :], 0) + 1e-6
            jmu = device_put(mu)
            jgamma = device_put(gamma)


    uc = jnp.reshape(z[:N], (o, -1))              # control input variables  #,
    epsilon = z[N:N+ncx*N]                        # slack variables on state constraints

    if status==0:
        if verbose:
            print("Converged to minima")
        x_check = simulate(xt, np.hstack([ut, uc]), w, theta)
        hx = jnp.concatenate([state_constraint(x_check[:, :, 1:]) for state_constraint in state_constraints], axis=2)
        cx = np.mean(hx > 0, axis=1)
        cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
        if (cx < (1-delta-1e-4)).any():
            if verbose:
                print("State constraints not satisfied with desired probability")
            status=1
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



## this is a version with slack constraints on inputs that is currently depreciated

def log_barrier_cost2(z, ut, xt, x_star, theta, w, sqc, src, delta, mu, gamma, simulate, state_constraints, input_constraints, N):

    print('Compiling log barrier cost')
    ncu = len(input_constraints)    # number of input constraints
    ncx = len(state_constraints)    # number of state constraints

    o = w.shape[0]
    uc = jnp.reshape(z[:N], (o, -1))                # control input variables  #,
    epsilon = z[N:N+ncx*N]                          # slack variables on state constraint probabilities
    s = z[N+ncx*N:N+ncx*N + ncu * N]                # slack variables on input constraints

    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    # state error and input penalty cost and cost that drives slack variables down
    V1 = jnp.sum(jnp.matmul(sqc,jnp.reshape(x - x_star,(o,-1))) ** 2) + jnp.sum(jnp.matmul(src, uc)**2) + jnp.sum(100 * (epsilon + 1e3)**2) +\
         jnp.sum(10.0 * (s + 1e3)**2)
    # need a log barrier on each of the slack variables to ensure they are positve
    V2 = logbarrier(epsilon - delta, mu) + logbarrier(s, mu)       # aiming for 1-delta% accuracy
    # now the chance constraints
    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma)
    cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
    V3 = logbarrier(cx, mu) + logbarrier(cu + s, mu)
    return V1 + V2 + V3

def solve_chance_logbarrier2(uc0, cost, gradient, hessian, ut, xt, theta, w, x_star, sqc, src, delta, simulate,
                            state_constraints, input_constraints, verbose=True, max_iter=1000, gamma=1.0, mu=1e4,
                            epsilon0=1.0, s0=10.0):

    [o,N] = uc0.shape

    ncu = len(input_constraints)  # number of input constraints
    ncx = len(state_constraints)  # number of state constraints

    z = np.hstack([uc0.flatten(), epsilon0*np.ones((ncx * N,)), s0 * np.ones((ncu * N,))])

    args = (device_put(ut), device_put(xt), device_put(x_star), device_put(theta),
            device_put(w), device_put(sqc), device_put(src), device_put(delta))

    jmu = device_put(mu)
    jgamma = device_put(gamma)

    for i in range(max_iter):
        # compute cost, gradient, and hessian
        jz = device_put(z)
        c = np.array(cost(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))
        g = np.array(gradient(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))
        h = np.array(hessian(jz, *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))

        # compute search direction
        p = - np.linalg.solve(h, g)
        # check that we have a valid search direction and if not then fix
        # TODO: make this less hacky (look at slides)
        beta2 = 1e-8
        while np.dot(p, g) > 0:
            p = - np.linalg.solve((h + beta2 * np.eye(h.shape[0])), g)
            beta2 = beta2 * 2

        # perform line search
        alpha = 1.0
        for k in range(52):
            # todo: (lower priority) need to use the wolfe conditions to ensure a bit of a better decrease
            ztest = z + alpha * p
            ctest = np.array(cost(device_put(ztest), *args, jmu, jgamma, simulate, state_constraints, input_constraints, N))
            # if np.isnan(ctest) or np.isinf(ctest):
            #     continue
            # nan and inf checks should be redundant
            if ctest < c:
                z = ztest
                break

            alpha = alpha / 2

        if k == 51 and np.abs(np.dot(g, p)) > 1e-2:
            status = 3
            print('Unable to find an alpha that decreases cost')
            break
        if verbose==2:
            print('Iter:', i + 1, 'Cost: ', c, 'nd:', np.dot(g, p), 'alpha: ', alpha, 'mu: ', mu, 'gamma: ', gamma)

        if np.abs(
                np.dot(g, p)) < 1e-2:  # if search direction was really small, then decrease mu and s for next iteration
            if mu < 1e-6 and gamma < 1e-3:
                status = 0
                break  # termination criteria satisfied

            mu = max(mu / 2, 0.999e-6)
            gamma = max(gamma / 1.25, 0.999e-3)
            # need to adjust the slack after changing gamma
            # x_new = simulate(xt, np.hstack([ut, row_vec(z[:N - 1])]), w, theta)
            x_new = simulate(xt, np.hstack([ut, row_vec(z[:N])]), w, theta)
            hx = jnp.concatenate([state_constraint(x_new[:, :, 1:]) for state_constraint in state_constraints], axis=2)
            # cx = chance_constraint(hx, z[N - 1:(N - 1) + ncx * (N - 1)], gamma)
            cx = chance_constraint(hx, z[N:N + ncx * N], gamma)
            z[N:N + ncx * N] += -np.minimum(cx[0, :], 0) + 1e-6
            jmu = device_put(mu)
            jgamma = device_put(gamma)


    uc = jnp.reshape(z[:N], (o, -1))              # control input variables  #,
    epsilon = z[N:N+ncx*N]                        # slack variables on state constraints
    s = z[N+ncx*N:N+ncx*N + ncu * N]

    if status==0:
        if verbose:
            print("Converged to minima")
        x_check = simulate(xt, np.hstack([ut, uc]), w, theta)
        hx = jnp.concatenate([state_constraint(x_check[:, :, 1:]) for state_constraint in state_constraints], axis=2)
        cx = np.mean(hx > 0, axis=1)
        cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1)
        if (cx < (1-delta-1e-4)).any():
            if verbose:
                print("State constraints not satisfied with desired probability")
            status=1
        if (cu < 1e-8).any():
            status=2
            if verbose:
                print("Input constraints violated")

    result = {'uc':uc,
              'epsilon':epsilon,
              's':s,
              'status':status,
              'gamma':gamma,
              'mu':mu,
              'gradient':g,
              'hessian':h,
              'newton_decrement':np.dot(g,p)}
    return result
