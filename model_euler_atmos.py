import numpy

from dune.common import FieldVector
from dune.grid import reader, cartesianDomain
from dune.fem.function import uflFunction
from ufl import *
from dune.ufl import cell, Constant


def model(dim=2, **kwargs):
    problem = ""
    cmpE = dim+1
    class Model:
        dimRange = dim+2
        cmpE = dim+1

        atmos = None
        g   = 9.80665 # gravity force [m/ss]
        T0  = 250.    # surface temperature [K]
        p0  = 100000. # surface pressure [Pa]
        c_p = 1005.   # todo: docme
        c_v = 717.95  # todo: docme
        N   = 0.01    # Brunt-Vaisala frequency (>0: stratified, 0: neutral, <0: isothermal atmosphere)
        mu  = Constant(0., "kinetic_viscosity")  # kinetic viscosity [mm/s]
        Re  = 1.      # Reynold's number
        Pr  = 1.      # Prandtl's number

        c_pd = 1005.  # specific heat of dry air at constant pressure, J/(kgK)=mm/(Kss)
        c_vd = 717.95 # specific heat of dry air at constant volume

        R_d = (c_pd - c_vd)
        kappa = R_d / c_pd
        gamma = (c_pd / c_vd )

        hSpeed0 = 0 # 20.

        # helper vars
        _gNcp = g*g/(N*N*c_p)
        _NNg  = N*N/g

        def bgPrim(t,x):
            # height in atmosphere
            z = x[dim-1]
            NNgz = Model._NNg * z

            ## N > 0, TODO, other cases
            # assert Model.N > 0, "Only N>0 implemented"
            assert Model.atmos is not None

            if Model.atmos == "stable":
                # assert Model.N > 0, "N does not fit to selected atmosphere"
                ## is this the 'stable atmosphere' from metstroem project?
                T = Model._gNcp + (Model.T0 - Model._gNcp) * exp( NNgz )
                p = Model.p0 * exp( 1./Model.kappa * (ln(T/Model.T0) - NNgz) )
                theta = T * pow( p/Model.p0, -Model.kappa )
            elif Model.atmos == "isothermal":
                # where is Model.N used in this atmosphere?
                # The derived quantities gNcp and NNgz are only used in
                # the 'stable' atmosphere as far as I could see.
                # What should Model.N be in the 'mountain' test case for example?
                # assert Model.N < 0, "N does not fit to selected atmosphere"
                T = Model.T0
                p = Model.p0*exp(-Model.g/Model.R_d/Model.T0*z)
                theta = T * pow( p/Model.p0, -Model.kappa )
            elif Model.atmos == "neutral":
                # assert Model.N == 0, "N does not fit to selected atmosphere"
                T = Model.T0 - z * Model.g / Model.c_p
                p = Model.p0 * pow( T / Model.T0, 1/Model.kappa )
                theta = Model.T0
            else:
                assert False, "unknown atmosphere"

            rho = p/T/Model.R_d # density
            u   = [ Model.hSpeed0] + [0]*(dim-1)
            BG = as_vector([rho, *u, theta])

            # return value of background atmos and p,T,theta
            return BG, p, T, theta, u

        # derivative of background atmosphere at (t,x) in primitive variables
        def jacBgPrim(t,x):

            assert Model.atmos == "neutral", "Not implemented for general atmosphere"
            # compute atmosphere
            _, p, T, _, _ = Model.bgPrim(t,x)

            dz_rho   = - (1./Model.R_d) * (1./Model.kappa - 1.) * Model.g / Model.c_pd * p / (T*T)
            dz_theta = Model.T0 * dz_rho

            jac = [ ([0]*dim) for i in range(dim+2)]
            jac[0][-1] = dz_rho
            jac[cmpE][-1] = dz_theta

            return as_vector(jac)

        # background atmosphere at (t,x)
        def bg(t,x):
            BG, p, T, theta, u = Model.bgPrim(t, x)
            return Model.toCons( BG ), p, T, theta, u

        # derivative background atmosphere at (t,x)
        def jacBg(t,x):
            return Model.jacBgPrim(t,x)

        # U --> V (conservative to primitive variables)
        def toPrim(U):
            rhoRm = Model.R_d * U[0]
            rhoCpml = Model.c_pd * U[0]

            kappaLoc = rhoRm / rhoCpml;
            p = Model.p0 * pow( Model.R_d * U[cmpE] / Model.p0, 1./(1.-kappaLoc) )
            T = p / rhoRm
            theta = U[cmpE] / U[0] # only for dry air
            v  = as_vector( [U[i]/U[0] for i in range(1,dim+1)] )
            return U[0], v, p, T, theta

        # V --> U (primitive to conservative variables)
        def toCons(V):
            m = as_vector( [V[i]*V[0] for i in range(1,dim+1)] )
            pT = V[cmpE]*V[0]
            return as_vector( [V[0],*m,pT] )

        # for flux computation of HLL_PT
        def pressureTemperature(U):
            _, _, p, T, _ = Model.toPrim(U)
            return as_vector([p, T])

        # internal source implementation
        def S_e(t,x,U,DU):
            S = [0.]*(len(U)-1)
            return as_vector([ 0,0,-Model.g * U[0], 0])

        # internal flux implementation
        def F_c(t,x,U):
            rho, v, p, _, theta = Model.toPrim(U)

            v = numpy.array(v)
            res = numpy.vstack([ rho*v,
                                 rho*numpy.outer(v,v) + p*numpy.eye(dim),
                                 (rho*theta)*v ])

            return as_matrix(res)

        # interface methods for model
        def bgFix(t,x,U):
            BG, p, T, theta, v = Model.bg(t,x)
            V = BG
            return as_vector(V)

        # interface method needed for LLF and time step control
        def maxWaveSpeed(t,x,U,n):
            rho, v, p, _, _ = Model.toPrim(U)
            return abs(dot(v,n)) + sqrt(Model.gamma*p/rho)
        # velocity of fluid
        def velocity(t,x,U):
            _, v ,_,_,_ = Model.toPrim(U)
            return v
        def physical(t,x,U):
            rho, _, p, _,_ = Model.toPrim(U)
            return conditional( rho>1e-8, conditional( p>1e-8 , 1, 0 ), 0 )
        def jump(t,x,U,V):
            _,_, pU, _,_ = Model.toPrim(U)
            _,_, pV, _,_ = Model.toPrim(V)
            return (pU - pV)/(0.5*(pU + pV))

        def noFlowFlux(u,n):
            _, _, p,_,_ = Model.toPrim(u)
            return as_vector([0]+[p*c for c in n]+[0])

        def rotateForth(u, n):
            if dim == 1:
                return [ u[0], n[0]*u[1], u[2] ]
            elif dim == 2:
                return [ u[0], n[0]*u[1] + n[1]*u[2], -n[1]*u[1] + n[0]*u[2], u[3] ]
            elif dim == 3:
                d = sqrt( n[0]*n[0]+n[1]*n[1] )
                d_1 = 1./d
                return [ u[0],
                         conditional(d>1e-9, n[0]*u[1] + n[1]*u[2] + n[2]*u[3], n[2] * u[3]),
                         conditional(d>1e-9,-n[1] * d_1 * u[1] + n[0] * d_1 * u[2], u[2]),
                         conditional(d>1e-9,- n[0] * n[2] * d_1 * u[1] - n[1] * n[2] * d_1 * u[2] + d * u[3], -n[2] * u[1]),
                         u[4] ]
        def rotateBack(u, n):
            if dim == 1:
                return [ u[0], n[0]*u[1], u[2] ]
            elif dim == 2:
                return [ u[0], n[0]*u[1] - n[1]*u[2],  n[1]*u[1] + n[0]*u[2], u[3] ]
            elif dim == 3: # assumption n[0]==0
                d = sqrt( n[0]*n[0]+n[1]*n[1] )
                d_1 = 1./d
                return [ u[0],
                         conditional(d>1e-9, n[0] * u[1] - n[1]*d_1 * u[2] - n[0]*n[2]*d_1 * u[3], -n[2]*u[3]),
                         conditional(d>1e-9, n[1] * u[1] + n[0]*d_1 * u[2] - n[1]*n[2]*d_1 * u[3], u[2]),
                         conditional(d>1e-9, n[2] * u[1] + d * u[3], n[2]*u[1]),
                         u[4] ]
        def reflect(u,n):
            uRot = Model.rotateForth(u, n)
            uRot[ 1 ] = -uRot[ 1 ]
            return as_vector( Model.rotateBack(uRot, n) )

        class Indicator:
            def eta(t,x,U):
                _,_, p,_,_ = Model.toPrim(U)
                return U[0]*ln(p/U[0]**Model.gamma)
            def F(t,x,U,DU):
                s = Model.Indicator.eta(t,x,U)
                _,v,_,_,_ = Model.toPrim(U)
                return v*s
            def S(t,x,U,DU):
                return 0

    return Model
