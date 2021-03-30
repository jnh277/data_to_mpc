## checkout https://github.com/stephane-caron/qpsolvers for a list of options

from numpy import array, dot
import numpy as np
from qpsolvers import solve_qp

# a demo using the qpsolvers interface

M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))
A = array([1., 1., 1.])
b = array([1.])
result1 = solve_qp(P, q, G, h, A, b)
print("QP solution:", result1)

# USING QUADPROG DIRECTLY, Requires the Hessian to be positive definite, this is a dense solver
# https://github.com/quadprog/quadprog
# Minimize     1/2 x^T G x - a^T x
# Subject to   C.T x >= b
# can also be used to solve equality constrained problems, look at above link
# documentation is pretty terrible

import quadprog


M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
G = dot(M.T, M)     # positive definite hessian
a = dot(M.T, array([3., 2., 3.]))
C = - array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]]).T
b = - array([3., 2., -2.]).reshape((3,))
meq = 0

result2 = quadprog.solve_qp(G,a,C,b,meq)

# using OSQP, sparse solver, but also quite fast on dense problems, handles semi definite case
# has warm starting and compiling options (looks really good)
# res.y is the dual variables (lagrange multipliers?)
# warm start prob.warm_start(x=x0, y=y0)
import osqp
from scipy import sparse

# Define problem data
P = sparse.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace and change alpha parameter
prob.setup(P, q, A, l, u, alpha=1.0, verbose=True)

# Solve problem
res = prob.solve()



