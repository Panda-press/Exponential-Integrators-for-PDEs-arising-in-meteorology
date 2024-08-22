import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import identity, diags

from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange, composite
from dune.fem import integrate
from dune.fem.operator import galerkin, molGalerkin
from ufl import ( TestFunction, TrialFunction, SpatialCoordinate, FacetNormal,
                  dx, ds, div, grad, dot, inner, exp, sin, conditional,
                  sin, pi, as_vector)

grid = leafGridView([0, 0], [1, 1], [100, 100])

epsilon = 1


space = lagrange(grid, order = 2)
x = SpatialCoordinate(space)
initial = sin(pi*x[0]) * sin(pi*x[1])
u_h = space.interpolate(0, name = "u_h")
u_h.interpolate(initial)
u_h.plot()
u_h_n = u_h.copy()

u = TrialFunction(space)
v = TestFunction(space)

a = epsilon**2*dot(grad(u), grad(v)) * dx

op = galerkin(a, rangeSpace=space, domainSpace=space)

dt = 0.1
T = 10000
t = 0
while t < T:
    u_h_n.assign(u_h)
    op(u_h, u_h_n)
    u_h.as_numpy[:] -= dt * u_h_n.as_numpy[:]
    t += dt
u_h.plot()