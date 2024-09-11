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
    Model.endTime = 400 # 1200.
    Model.name = "RisingBubble"

    return BgFixModel(Model, dim)

import sys
import numpy as np
import matplotlib.pyplot as plt
from dune.grid import structuredGrid
from dune.fem.space import dglagrange, finiteVolume
from dune.femdg import femDGModels, femDGOperator, advectionNumericalFlux
from dune.femdg.rk import femdgStepper
from steppers import steppersDict

if __name__ == "__main__":
    stepperFct, args = steppersDict[sys.argv[1]]
    if len(sys.argv) >= 3:
        cfl = float(sys.argv[2])
        outputName = lambda n: f"risingbubble_{sys.argv[1]}{cfl}_{n}.png"
    else:
        cfl = 0.45
        outputName = lambda n: f"risingbubble_{sys.argv[1]}default_{n}.png"

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

    if len(sys.argv) >= 4:
        if "exp_v" in args.keys():
            m = int(sys.argv[3])
            args["expv_args"] = {"m":m}

    stepper = stepperFct(op, mass='identity', **args)

    # get initial time step size - just using some very small timestep
    info = stepper(target=u_h, tau=1e-5)
    u_h.interpolate(Model.U0)

    # time loop
    # figure out a first tau

    t = 0
    n = 0
    fileCount = 0
    plotTime = 1
    nextTime = 1

    u_h.plot(gridLines=None, block=False)
    plt.savefig(outputName(fileCount))
    fileCount += 1
    lastNcalls = op.info()[0]

    while t < Model.endTime:
        op.setTime(t)
        tau = op.localTimeStepEstimate[0]*cfl
        info = stepper(target=u_h, tau=tau)
        t += tau

        assert not np.isnan(u_h.as_numpy).any()
        n += 1
        if True: # t>plotTime:
            minMax = max(abs(u_h.as_numpy))
            print(f"time step {n}, time {t}, tau {tau}, calls {op.info()}, lastNcalls {op.info()[0]-lastNcalls}, minMax={minMax}")
            lastNcalls = op.info()[0]
            u_h.plot(gridLines=None, block=False)
            plt.savefig(outputName(fileCount))
            fileCount += 1
            plt.close()
            plotTime += nextTime

    plt.savefig(outputName(fileCount))
