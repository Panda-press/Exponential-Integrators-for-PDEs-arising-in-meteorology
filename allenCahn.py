import numpy as np
from ufl import *
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [0, 0], [1, 1], [60, 60]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

eps = Constant(0.01,"eps")
theta = Constant(0.8,"theta")
theta_c = Constant(1.6,"theta_c")

"""
a = ( eps**2 * inner(grad(u),grad(v))
      - theta/2 * ln( (1-u[0])/(1+u[0]) ) * v[0]
      - theta_c * dot(u,v)
    ) * dx
"""

a = ( eps**2 * inner(grad(u),grad(v))
      + (dot(u,u)-1) * dot(u,v)
    ) * dx

"""
def test1(gridView):
    h = 0.01
    g0 = lambda x,x0,T: conditional(x-x0<-T/2,0,conditional(x-x0>T/2,0,sin(2*pi/T*(x-x0))**3))
    G  = lambda x,y,x0,y0,T: g0(x,x0,T)*g0(y,y0,T)
    return a, 20, 0.9*G(x[0],x[1],0.5,0.5,50*h), None
"""

def test2(gridView):
    tauFE = 0.8   # AC on level=0
    @gridFunction(gridView,name="random",order=1)
    def u0(x):
        return 1.8*np.random.rand(1)[0]-0.9
    np.random.seed(100)
    return a, 24, tauFE, u0, None
