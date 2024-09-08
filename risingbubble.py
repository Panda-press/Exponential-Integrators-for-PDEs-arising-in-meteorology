import sys

from dune.fem.function import gridFunction
from dune.ufl import cell, Constant
from ufl import SpatialCoordinate, sqrt, exp, sin, pi, conditional, as_vector

from model_euler_atmos import model as AtmosphericModel
from model_bgfix import model as BgFixModel


def RisingBubble(dim=2):
    Model = AtmosphericModel(dim=dim)
    Model.atmos = "neutral"
    Model.hSpeed0 = 0.
    Model.T0 = 303.15
    #Model.K = 0.1

    x = SpatialCoordinate(cell(dim))
    Model.dx = Constant(0., name="dx")

    z   = x[dim-1] # z-coordinate
    #x_c = 533.     # x-center of perturbation ball
    x_c = 500. + 0.5*Model.dx  # x-center of perturbation ball
    z_c = 520.     # z-center of perturbation ball
    a   = 50       # radius of perturbation ball

    r = sqrt((x[0] - x_c)**2 + (z - z_c)**2)

    BG, p, T, theta, v = Model.bg(0, x)

    A = 0.5
    s = 100.

    deltaTheta = A * conditional(r <= a,
        1.0,
        exp( -(r-a)**2/s**2)
    )
    newTheta = theta + deltaTheta

    # introduce pot. temp. to conservative variables keeping
    # the pressure equal to the background pressure

    rho = pow( Model.p0, Model.kappa ) * pow( p, 1./Model.gamma ) / Model.R_d / newTheta

    Model.U0 = Model.toCons(as_vector([rho] + v + [newTheta]))

    Model.boundary = {1: lambda t,x,U: Model.reflect(U,[-1, 0]),
                      2: lambda t,x,U: Model.reflect(U,[ 1, 0]),
                      3: lambda t,x,U: Model.reflect(U,[ 0,-1]),
                      4: lambda t,x,U: Model.reflect(U,[ 0, 1])}


    Model.domain = [0]*dim, [1000]*(dim-1)+[2000], [40]*(dim-1)+[80]
    Model.endTime = 1200.
    Model.name = "RisingBubble"

    return BgFixModel(Model, dim)

import sys
import numpy as np
import matplotlib.pyplot as plt
from dune.grid import structuredGrid
from dune.fem.space import dglagrange, finiteVolume
from dune.femdg import femDGModels, femDGOperator, advectionNumericalFlux
from dune.femdg.rk import femdgStepper
from steppers import BEStepper, FEStepper, ExponentialStepper, SIStepper
from steppers import expm_kiops, expm_lanzcos, expm_nbla, expm_arnoldi, expm_multiply

if __name__ == "__main__":
    steppersDict = {"FE": (FEStepper,{}),
                    "BE": (BEStepper,{"ftol":1e-3,"verbose":False}),
                    "SI": (SIStepper,{}),
                    "EXPSCI": (ExponentialStepper, {"exp_v":expm_multiply}),
                    "EXPLAN": (ExponentialStepper, {"exp_v":expm_lanzcos}),
                    "EXPNBLA": (ExponentialStepper, {"exp_v":expm_nbla}),
                    "EXPARN": (ExponentialStepper, {"exp_v":expm_arnoldi}),
                    "EXPKIOPS": (ExponentialStepper, {"exp_v":expm_kiops}),
                   }
    stepperFct, args = steppersDict[sys.argv[1]]
    if len(sys.argv) > 2:
        cfl = int(sys.argv[2])
        outputName = lambda n: f"risingbubble_{sys.argv[1]}{cfl}_{n}"
    else:
        cfl = 0.45
        outputName = lambda n: f"risingbubble_{sys.argv[1]}default_{n}"

    # default name for model
    Model = RisingBubble(2)
    gridView = structuredGrid( *Model.domain )
    gridView.hierarchicalGrid.globalRefine(3)

    space = finiteVolume(gridView,dimRange=Model.dimRange)
    # space = dglagrange(gridView,dimRange=Model.dimRange,order=3,pointType="lobatto")
    u_h = space.interpolate(Model.U0, name="solution")

    models = femDGModels(Model,space)

    operator_kwargs = dict(limiter=None,
                      advectionFlux="Dune::Fem::AdvectionFlux::Enum::euler_hllc_bgfix",
                      codegen=False,
                      threading=True,
                      defaultQuadrature=True)
    op = femDGOperator(models, space, **operator_kwargs)

    stepper = stepperFct(op, mass='identity', **args)

    # get initial time step size
    info = stepper(target=u_h, tau=1e-5)
    u_h.interpolate(Model.U0)

    # stepper = femdgStepper(order=1, operator=op)

    # time loop
    # figure out a first tau

    t = 0
    n = 0
    plotTime = 0.5
    nextTime = 0.5

    u_h.plot(gridLines=None, block=False)
    plt.savefig(outputName(n))

    while t < 400:
        op.setTime(t)
        tau = op.localTimeStepEstimate[0]*cfl

        """
        t += stepper(u_h)
        """
        info = stepper(target=u_h, tau=tau)
        t += tau

        assert not np.isnan(u_h.as_numpy).any()
        n += 1
        print(f"time step {n}, time {t}, tau {tau}, calls {op.info()}")
        if t>plotTime:
            u_h.plot(gridLines=None, block=False)
            plt.savefig(outputName(n))
            plt.close()
            plotTime += nextTime
