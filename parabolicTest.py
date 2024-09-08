from ufl import *
from dune.ufl import Space, Constant

dimR = 1
time = Constant(0,"time")
sourceTime = Constant(0,"sourceTime")
domain = [0, 0], [1, 1], [80, 80]

space = Space(2,dimRange=dimR)
x,u,v,n = ( SpatialCoordinate(space), TrialFunction(space), TestFunction(space), FacetNormal(space) )

def model(exact,dtExact,nonLinear):
    a  = ( inner(grad(u),grad(v)) + dot(nonLinear(u),v) ) * dx
    bf = lambda t: ( ( dtExact(t)[0] - div(grad(exact(t)[0])) ) * v[0]
                     + dot(nonLinear(exact(t)),v) 
                   ) * dx 
    bg = lambda t: dot(grad(exact(t)[0]),n)*v[0]*ds
    return a-bf(sourceTime)-bg(sourceTime)

def paraTest0(gridView):
    freq = 2*pi
    xExact = as_vector([ cos(freq*x[0]) ])
    exact = lambda t: exp(-freq**2*t) * xExact
    dtExact = lambda t: -freq**2*exp(-freq**2*t) * xExact
    return model(exact,dtExact, lambda u: as_vector([0])), 0.5, exact(time), exact

def paraTest1(gridView):
    freq = 6*pi
    xExact = as_vector([ cos(freq*x[0]) ])
    exact = lambda t: exp(-freq**2*t) * xExact
    dtExact = lambda t: -freq**2*exp(-freq**2*t) * xExact
    return model(exact,dtExact, lambda u: (1+dot(u,u))*u), 0.5, exact(0), exact

def paraTest2(gridView):
    freq = 2*pi
    xExact = as_vector([ cos(freq*x[0]) ])
    exact = lambda t: exp(-freq**2*t) * xExact
    dtExact = lambda t: -freq**2*exp(-freq**2*t) * xExact
    return model(exact,dtExact, lambda u: (1+dot(u,u))*u), 0.5, exact(time), exact
