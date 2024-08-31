# %% [markdown]
#
# # Exponential Integrators
#
# ## Setup

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse import identity, diags

from dune.grid import structuredGrid as leafGridView
from dune.fem.space import lagrange
from dune.fem import integrate
from dune.fem.operator import galerkin
from ufl import ( TestFunction, TrialFunction, SpatialCoordinate, FacetNormal,
                  dx, ds, div, grad, dot, inner, conditional, as_vector,
                  exp, sin, pi )

from scipy.sparse.linalg import expm_multiply
# %% [markdown]
#
# Simple utility function to show the result of a simulation

# %%
def printResult(method,error):
    print(method,"L^2, H^1 error:",'{:0.5e}, {:0.5e}'.format(
                 *[ np.sqrt(e) for e in integrate([error**2,inner(grad(error),grad(error))]) ]),
          flush=True)

# %% [markdown]
# ## A base class for the steppers
#
# Assume the non-linear operator is given by `N` so we are solving
# $u_t + N[u] = 0$
# 

# %%
class BaseStepper:
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
        self.countN = 0
        self.linIter = 0
        # time step to use
        self.tau = None

        if not "expl" in self.method:     # for all non-explicit method we need the linearization
            self.A = self.N.linear()
            self.d = self.un.copy()       # search direction for non-linear method
            self.I = identity(self.shape[0])
            self.D = None                 # store 'implicit Euler' operator D = I + tau M^{-1}DN[u]

    # call before computing the next time step
    def setup(self,un,tau):
        self.un.assign(un)
        self.tau = tau
        if self.method == "quasiNewton":
            self.D = self.matrix(self.un) # we only compute 'D' once 

    # evaluate w = tau M^{-1} N[u]
    def evalN(self,u,w):
        self.N(u, w)
        w.as_numpy[:] *= self.tau * self.Minv
        self.countN += 1

    # linearize the equation around current stage (or use fixed 'D')
    def linearize(self,u):
        assert not self.method == "explicit"
        if self.method == "quasiNewton":
            pass
        elif self.method == "Newton":
            self.D = self.matrix(u)
        elif self.method == "approxDN":
            self.D = self.DNapprox(self, u)

    # compute matrix D = tau M^{-1}DN[u] + I
    # e.g. BE: u^{n+1} + tau M^{-1}N[u^{n+1}] = u^n
    # lineaaized around u^n_k: (I + tau M^{-1}DN[u^n_k])u = u^n
    def matrix(self,u):
        self.N.jacobian(u,self.A)
        D = self.tau * self.Minv @ self.A.as_numpy
        D += self.I
        return D

    # _matvec(x) returns ( tau M^{-1}DN(u) + I )x approximated by
    #            tau M^{-1} ( N(u + eps x) - N(u) ) / eps + x
    class DNapprox(LinearOperator):
        def __init__(self,stepper, u):
            self.stepper = stepper
            self.u = u.copy()
            self.Nu = u.copy()
            self.uplusx = u.copy()
            self.Nuplusx = u.copy()

            self.norm_u = u.as_numpy.dot(u.as_numpy)
            self.stepper.evalN(u,self.Nu)

            self.shape = stepper.shape
            self.dtype = np.dtype(np.float64)
        def epsilon(self,x):
            meps = np.finfo(np.float64).eps
            xdotx = x.dot(x)
            if xdotx > meps:
                return np.sqrt( ( 1 + self.norm_u) * meps / xdotx )
            else:
                return np.sqrt( meps )
        def _matvec(self,x):
            eps = self.epsilon(x)
            # u + eps x
            self.uplusx.as_numpy[:] = self.u.as_numpy[:] + eps * x[:]
            # N(u + eps x)
            self.stepper.evalN(self.uplusx,self.Nuplusx)
            ret = (self.Nuplusx.as_numpy[:] - self.Nu.as_numpy[:]) / eps
            ret[:] += x
            return ret

    def callback(self,x):
        self.linIter += 1

# %% [markdown]
# ## Forward Euler method

# %%
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau N(u^n)
class FEStepper(BaseStepper):
    def __init__(self, N):
        BaseStepper.__init__(self,N,"explicit")
        self.name = "FE"

    def __call__(self, target, tau):
        self.setup(target,tau)
        self.evalN(self.un,self.res)
        target.as_numpy[:] -= self.res.as_numpy[:]
        return {"iterations":1, "linIter":0}

# %% [markdown]
# ## Backward Euler method

# %%
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau N(u^{n+1})
class BEStepper(BaseStepper):
    def __init__(self, N, method):
        BaseStepper.__init__(self,N,method)
        self.name = "BE"

    def __call__(self, target, tau):
        self.setup(target,tau)
        # get numpy vectors for target, residual, search direction
        sol_coeff = target.as_numpy
        res_coeff = self.res.as_numpy
        d_coeff   = self.d.as_numpy
        # now implement a Newton method to compute root u^{n+1} of
        # F(u) = (u - u^n) + tau M^{-1}N(u)
        # So
        # 1) r = F(u^{n+1,k}) = (u^{n+1,k} - u^n) + tau M^{-1}N(u^{n+1,k})
        # 2) D = DF(u^{n+1,k} = M^{-1}DN(u^{n+1},k}) 
        # 3) Solve Dd = r
        # 4) u^{n+1,k+1} = u^{n+1,k} - d
        k = 0
        self.linIter = 0
        while True:
            self.evalN(target,self.res)
            res_coeff[:] += ( sol_coeff - self.un.as_numpy )
            # check for convergence
            absF = np.sqrt( np.dot(res_coeff,res_coeff) )
            print(f"iter {k} : {absF} < {1e-6} {self.linIter}",flush=True)
            if absF < 1e-6:
                break
            self.linearize(target)
            d_coeff[:],_ = cg(self.D, res_coeff, callback=lambda x: self.callback(x) ) # note that this assumes a symmetric D
            sol_coeff[:] -= d_coeff[:]
            k += 1
        return {"iterations":k, "linIter":self.linIter}

# %% [markdown]
# ## A semi-implicit approach (with a linear implicit part)

# %%
# solve d_t u + N(u) = 0 using u^{n+1} = u^n - tau A^n u^{n+1} - tau R^n(u^n)
# with A^n = DN(u^n) and R^n(u^n) = N(u^n) - A^n u^n, i.e., rewritting
# N(u) = DN(u)u + (N(u) - DN(u)u)
# So in each step we need to solve
# ( I + tau A^n ) u^{n+1} = u^n - tau R^n(u^n)
#                         = ( I + tau A^n) u^n - tau N(u^n)
# u^{n+1} = u^n - tau ( I + tau A^n )^{-1} N(u^n)
class SIStepper(BaseStepper):
    def __init__(self, N):
        BaseStepper.__init__(self,N, "quasiNewton")  # quasiNewton computes D in 'setup'
        self.name = "SI1"

    def explStep(self, target, tau):
        # this sets D = I + tau A^n and initialized u^n
        self.setup(target, tau)
        # get numpy vectors for target, residual, search direction
        sol_coeff = target.as_numpy
        res_coeff = self.res.as_numpy

        # compute N(u^n)
        self.evalN(target,self.res)
        # subtract Du^n
        res_coeff[:] -= self.D@sol_coeff
        res_coeff[:] *= -1

    def __call__(self, target, tau):
        self.explStep(target,tau)

        sol_coeff = target.as_numpy
        res_coeff = self.res.as_numpy
        self.linIter = 0
        # solve linear system
        sol_coeff[:],_ = cg(self.D, res_coeff, callback=lambda x: self.callback(x) ) # note that this assumes a symmetric D
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
    def __init__(self, N, method=expm_multiply, method_name = ""):
        SIStepper.__init__(self,N)
        self.name = "ExpInt{0}".format(method_name)
        self.exp_v = method

    def __call__(self, target, tau):
        # u^* = (I + tau A^n)u^n - tau N(u^n)
        # Note: the call method on the base class calls 'setup' which computes the right A^n
        self.explStep(target,tau)
        sol_coeff = target.as_numpy
        res_coeff = self.res.as_numpy
        self.linIter = 0

        # Compute e^{-tau A^n}u^*
        sol_coeff[:] = self.exp_v(-self.Minv @ self.A.as_numpy * tau, res_coeff[:])
        
        return {"iterations":0, "linIter":self.linIter}


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

exact = as_vector([ ( 1/2*(x[0]**2+x[1]**2) - 1/3*(x[0]**3 - x[1]**3) + 1 ) *
                    ( 1.5+sin(5*pi*(x[0]+x[1])) ) ])

# f = - laplace u      and grad(u).n = grad(exact).n on the boundary and f = - laplace(exact)
# fv dx = - laplace u v dx = grad(u).grad(v) dx - grad(u).n v ds
# grad(u).grad(v) dx - (-laplace exact)v dx - grad(exact).n v ds = 0

a  = ( inner(grad(u),grad(v)) ) * dx # + (1+dot(u,u))*dot(u,v) ) * dx
bf = (-div(grad(exact[0])) ) * v[0] * dx # + (1+exact[0]**2)*exact[0]) * v[0] * dx
bg = dot(grad(exact[0]),n)*v[0]*ds
op = galerkin(a-bf-bg)

# %% [markdown]
#
# ## Time loop
#

# %%
tauFE = 7e-5 # time step (FE fails with tau=8e-5 on the [80,80] grid)

if False: # use FE
    # stepper = FEStepper(op)
    tau = tauFE
else:
    stepper = BEStepper(op, method="Newton")
    # stepper = SIStepper(op)
    # stepper = ExponentialStepper(op, method=expm_multiply, method_name="Scipy")
    tau = tauFE*100 # *100 # *10000

# initial condition
u_h.interpolate( exact )
upd = u_h.copy()

# time loop
time = 0
n = 0
totalIter, linIter = 0, 0
run = []
check = 1e-2
while True:
    upd.assign(u_h)
    info = stepper(target=u_h, tau=tau)
    assert not np.isnan(u_h.as_numpy).any()
    time += tau
    totalIter += info["iterations"]
    linIter += info["linIter"]
    # we expect u^n -> exact for n->infty, i.e., u' -> 0
    # so we stop if upd = u^n - u^{n+1} is small
    upd.as_numpy[:] -= u_h.as_numpy[:]
    upd.plot()
    update = np.sqrt( np.dot(upd.as_numpy,upd.as_numpy) )
    print(f"time step {n}, time {time}, N {stepper.countN}, iterations {info}, update {update}")
    if update < check:
        run += [(check,stepper.countN,linIter)]
        check /= 10
    if update < 1e-7:
        break
    n += 1
printResult(f"{stepper.name}{(stepper.countN,totalIter,linIter)}",u_h-exact)
print(run)
u_h.plot()
