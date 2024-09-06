# %% [markdown]
#
# # Exponential Integrators
#
# ## Setup

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import lgmres, gmres, LinearOperator, aslinearoperator
from scipy.sparse import identity, diags
from scipy.optimize import newton_krylov, KrylovJacobian
from scipy.sparse.linalg import expm_multiply
import random

from Stable_Lanzcos import LanzcosExp
expm_lanzcos = lambda A,x: LanzcosExp(A,x,5)

from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange
from dune.fem import integrate
from dune.fem.operator import galerkin
from dune.ufl import Constant
from ufl import ( TestFunction, TrialFunction, SpatialCoordinate, FacetNormal,
                  dx, ds, div, grad, dot, inner, conditional, as_vector,
                  exp, sin, cos, pi )

# %% [markdown]
#
# Simple utility function to show the result of a simulation

# %%
def printResult(time,error,*args):
    print(time,'{:0.5e}, {:0.5e}'.format(
                 *[ np.sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]),
          *args, " # gnuplot", flush=True)

# %% [markdown]
# ## A base class for the steppers
#
# Assume the non-linear operator is given by `N` so we are solving
# $u_t + N[u] = 0$
# 

# %%
class BaseStepper:
    # method is 'explicit', 'approx', 'exact'
    def __init__(self, N, method="explicit"):
        self.N = N
        self.method = method
        self.spc = N.domainSpace
        self.un = self.spc.zero.copy()  # previous time step
        self.res = self.un.copy()       # residual
        self.shape = (self.spc.size,self.spc.size)

        # inverse lumped mass matrix (see the explanation given in 'wave.py' tutorial example)
        # u^n v dx = sum_j u^n_j phi_j phi_i dx = M u^n
        # Mu^{n+1} = Mu^n - dt N(u^n) (for FE)
        # u^{n+1} = u^n - dt M^{-1}N(u^n)
        M = galerkin(dot(u,v)*dx).linear().as_numpy
        Mdiag = M.sum(axis=1) # sum up the entries onto the diagonal
        self.Minv = diags( 1/Mdiag.A1, shape=self.shape )

        # time step to use
        self.tau = None

        if self.method == "exact":
            self.Nprime = self.N.linear()

        self.I = identity(self.shape[0])
        self.tmp = self.un.copy()
        self.countN = 0
        self.linIter = 0

    # call before computing the next time step
    def setup(self,un,tau):
        self.un.assign(un)
        self.tau = tau
        if not "expl" in self.method:
            self.linearize(self.un)

    # evaluate w = tau M^{-1} N[u]
    def evalN(self,x):
        xh = self.N.domainSpace.function("tmp", dofVector=x)
        self.N(xh, self.tmp)
        self.countN += 1
        return self.tmp.as_numpy * (self.tau * self.Minv)

    # compute matrix D = tau M^{-1}DN[u] + I
    # e.g. BE: u^{n+1} + tau M^{-1}N[u^{n+1}] = u^n
    # lineaaized around u^n_k: (I + tau M^{-1}DN[u^n_k])u = u^n
    def linearize(self,u):
        assert not self.method == "expl"
        if self.method == "approx":
            self.A = aslinearoperator(self.Aapprox(self, u))
            self.D = aslinearoperator(self.Dapprox(self, u))
            # self.test(u)
        else:
            assert False # we are not considering 'exact' at the moment
            self.N.jacobian(u,self.Nprime)
            self.A = self.tau * self.Minv @ self.Nprime.as_numpy
            self.D = self.A + self.I

    def test(self,u):
        self.N.jacobian(u,self.Nprime)
        self.A_ = self.tau * self.Minv @ self.Nprime.as_numpy
        x = 0*self.tmp.as_numpy
        maxVal = 0
        for k in range(10):
            i = random.randrange(0,len(x),1)
            x[i] = random.random()
            y = self.A@x - self.A_@x
            maxVal = max(maxVal,y.dot(y))
            # print(i,maxVal)
        print("difference:",maxVal)

    class Aapprox:
        def __init__(self,stepper, u):
            self.krylovJacobian = KrylovJacobian()
            self.shape = (u.as_numpy.shape[0],u.as_numpy.shape[0])
            self.dtype = u.as_numpy.dtype
            f = stepper.evalN(u.as_numpy)
            self.krylovJacobian.setup(u.as_numpy, f, stepper.evalN)
        # issue with x having shape (N,1) needed by exponential method and
        # not handled by KrylovJacobian
        def matvec(self,x):
            if len(x.shape) > 1:
                y = x.copy()
                y[:,0] = self.krylovJacobian.matvec(x[:,0])
            else:
                y = self.krylovJacobian.matvec(x)
            return y
        # assume problem is self-adjoint - how does this hold for
        # the linearization and needs fixing for non self-adjoint PDEs
        def rmatvec(self,x):
            return self.matvec(x)
        def solve(self, rhs, tol=0):
            return self.krylovJacobian.solve(rhs,tol)

    class Dapprox:
        def __init__(self,stepper, u):
            self.krylovJacobian = KrylovJacobian()
            self.stepper = stepper
            self.shape = (u.as_numpy.shape[0],u.as_numpy.shape[0])
            self.dtype = u.as_numpy.dtype
            f = stepper.evalN(u.as_numpy)
            self.krylovJacobian.setup(u.as_numpy, f, stepper.evalN)
        # issue with x having shape (N,1) needed by exponential method and
        # not handled by KrylovJacobian
        def matvec(self,x):
            if len(x.shape) > 1:
                y = x.copy()
                y[:,0] = self.krylovJacobian.matvec(x[:,0]) + x
            else:
                y = self.krylovJacobian.matvec(x) + x
            return y
        # assume problem is self-adjoint - how does this hold for
        # the linearization and needs fixing for non self-adjoint PDEs
        def rmatvec(self,x):
            return self.matvec(x)

    # the below should be the same - need to test to make sure
    """
    class Dapprox:
        def __init__(self,A):
            self.shape = A.shape
            self.dtype = A.dtype
            self.A = A
        def matvec(self,x):
            y = self.A@x
            print(y.dot(y))
            return self.A@x + x
    """

# %% [markdown]
# ## Forward Euler method
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau N(u^n)
class FEStepper(BaseStepper):
    def __init__(self, N):
        BaseStepper.__init__(self,N,"explicit")
        self.name = "FE"

    def __call__(self, target, tau):
        self.setup(target,tau)
        target.as_numpy[:] -= self.evalN(self.un.as_numpy[:])
        return {"iterations":1, "linIter":0}

# %% [markdown]
# ## Backward Euler method
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau N(u^{n+1})
class BEStepper(BaseStepper):
    def __init__(self, N, method="approx"):
        BaseStepper.__init__(self,N,method)
        self.name = f"BE({method})"

    # non linear function (non-linear problem want f(x)=0)
    def f(self, x):
        return self.evalN(x) + ( x - self.un.as_numpy )
    def callback(self,x): 
        y = self.f(x)
        y = self.D@x - self.res.as_numpy
        print(self.linIter,y.dot(y), self.countN)
        self.linIter += 1


    def __call__(self, target, tau):
        self.setup(target,tau)
        # get numpy vectors for target, residual, search direction
        sol_coeff = target.as_numpy
        sol_coeff[:] = newton_krylov(self.f, xin=sol_coeff, f_tol=1e-6)
        return {"iterations":0, "linIter":self.linIter}

# %% [markdown]
# ## A semi-implicit approach (with a linear implicit part)
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau A^n u^{n+1} - tau R^n(u^n)
# with A^n = DN(u^n) and R^n(u^n) = N(u^n) - A^n u^n, i.e., rewritting
# N(u) = DN(u)u + (N(u) - DN(u)u)
# So in each step we need to solve
# ( I + tau A^n ) u^{n+1} = u^n - tau R^n(u^n)
#                         = ( I + tau A^n ) u^n - tau N(u^n)
# u^{n+1} = u^n - tau ( I + tau A^n )^{-1} N(u^n)
class SIStepper(BaseStepper):
    def __init__(self, N, method="approx"):
        BaseStepper.__init__(self,N, method)  # quasiNewton computes D in 'setup'
        self.name = f"SI({method})"

    def explStep(self, target, tau):
        # this sets D = I + tau A^n and initialized u^n
        self.setup(target, tau)
        res_coeff = self.res.as_numpy
        # compute N(u^n)
        res_coeff[:] = self.evalN(target.as_numpy)
        # subtract Du^n
        res_coeff[:] -= self.D@target.as_numpy
        res_coeff[:] *= -1

    def callback(self,x): 
        y = self.D@x - self.res.as_numpy
        print(self.linIter,y.dot(y), self.countN)
        self.linIter += 1

    def __call__(self, target, tau):
        self.explStep(target, tau)

        sol_coeff = target.as_numpy
        res_coeff = self.res.as_numpy
        self.linIter = 0
        # solve linear system
        sol_coeff[:],_ = lgmres(self.D, res_coeff, # x0=self.un.as_numpy,
                                rtol=1e-4,
                                # callback=lambda x: self.callback(x),
                                # callback_type='x'
                               )
        return {"iterations":0, "linIter":self.linIter}

# %%
# solve d_t u + N(u) = 0 using exponential integrator for d_t u + A^n u + R^n(u) = 0
# with A^n = DN(u^n) and R^n(u) = N(u) - A^n u
# Set v = e^{A^n(t-t^n)} u then
# d_t v = e^{A^n(t-t^n)) A^n u + e^{A^n(t-t^n)} d_t u
#       = e^{A^n(t-t^n)) A^n u - e^{A^n(t-t^n)} (A^n u + R^n(u))
#       = - e^{A^n(t-t^n)) R^n(u)
#       = - e^{A^n(t-t^n)) R^n( e^{-A^n(t-t^n)}v )
# Then using FE:
# v^{n+1} = v^n - tau R^n(v^n)
# e^{A^n tau)u^{n+1} = u^n - tau R^n(u^n) since u(t^n) = v(t^n)
# u^{n+1} = e^{-A^n tau} ( u^n - tau R^n(u^n) )
#         = e^{-A^n tau} ( u^n - tau (N(u^n) - A^n u^n) )
#         = e^{-A^n tau} ( (I + tau A^n)u^n - tau N(u^n) )
class ExponentialStepper(SIStepper):
    def __init__(self, N, exp_v, method_name, method='approx'):
        SIStepper.__init__(self,N,method)
        self.name = f"ExpInt({method},{method_name})"
        self.exp_v = exp_v

    def __call__(self, target, tau):
        # u^* = (I + tau A^n)u^n - tau N(u^n)
        # Note: the call method on the base class calls 'setup' which computes the right A^n
        self.explStep(target,tau)
        # Compute e^{-tau A^n}u^*
        target.as_numpy[:] = self.exp_v(- self.A, self.res.as_numpy)
        return {"iterations":0, "linIter":0}


# %% [markdown]
#
# # Numerical experiment
#
# ## Setup grid and space
#

# %%
dimR = 1
order = 1
domain = [0, 0], [1, 1], [80, 80]

gridView = leafGridView(*domain)
space = lagrange(gridView, order=order, dimRange=dimR)
u_h = space.interpolate(0, name='u_h')

# %% [markdown]
#
# ## Setup operator
#

# %%
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")

xExact = as_vector([ cos(2*pi*x[0]) ])
exact = lambda t: exp(-(2*pi)**2*t)*xExact
dtExact = lambda t: -(2*pi)**2*exp(-(2*pi)**2*t)*xExact

# exact = lambda t: cos(2*pi*t)*xExact
# dtExact = lambda t: -2*pi*sin(2*pi*t)*xExact

a  = ( inner(grad(u),grad(v)) + (1+dot(u,u))*dot(u,v) ) * dx
bf = lambda t: ( dtExact(t)[0] - div(grad(exact(t)[0]))
                 + (1+exact(t)[0]**2)*exact(t)[0]
               ) * v[0] * dx
bg = lambda t: dot(grad(exact(t)[0]),n)*v[0]*ds
op = galerkin(a-bf(sourceTime)-bg(sourceTime))

# %% [markdown]
#
# ## Time loop
#

# %%
factor = 100
tauFE = 7e-5 # time step (FE fails with tau=8e-5 on the [80,80] grid)

steppersDict = {"FE": (FEStepper(op), tauFE),
                "BE": (BEStepper(op), tauFE*factor),
                "SI": (SIStepper(op), tauFE*factor),
                "EXPSCI": (ExponentialStepper(op,
                           exp_v=expm_multiply, method_name="Scipy"),
                           tauFE*factor),
                "EXPLAN": (ExponentialStepper(op,
                           exp_v=expm_lanzcos, method_name="Lanzcos"),
                           tauFE*factor),
               }
stepper, tau = steppersDict[sys.argv[1]]

# initial condition
u_h.interpolate( exact(time) )

# time loop
n = 0
totalIter, linIter = 0, 0
run = []
checkTime = factor*tauFE
printResult(time.value,u_h-exact(time))
while time.value < 0.5:
    # this actually depends probably on the method we use, i.e., BE would
    # be + tau and the others without
    sourceTime.value = time.value # + tau
    info = stepper(target=u_h, tau=tau)
    assert not np.isnan(u_h.as_numpy).any()
    time.value += tau
    totalIter += info["iterations"]
    linIter   += info["linIter"]
    n += 1
    if time.value >= checkTime:
        print(f"time step {n}, time {time.value}, N {stepper.countN}, iterations {info}")
        printResult(time.value,u_h-exact(time),stepper.countN)
        run += [(stepper.countN,linIter)]
        checkTime += factor*tauFE
        # u_h.plot()
