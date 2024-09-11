from inspect import signature

import numpy

from dune.ufl import cell
from ufl import *
from dune.common import FieldVector, comm
from dune.grid import reader, cartesianDomain
from dune.fem.function import uflFunction


def parprint(*args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs, flush=True)


def model(Model, dim=2, **kwargs):
    x = SpatialCoordinate(cell(dim))

    class BGModel(Model):

        # overloaded source applying bgFix

        def consBG(U):
            BG, _, _, _, _ = Model.bg(0,x)
            return U + BG
        def toPrimBG(U):
            return as_vector([Model.toPrim(consBG(U))])
        def thetaTilde(U):
            BG, _, _, thetaBG, _ = Model.bg(0,x)
            _,  _, _, _, theta   = Model.toPrim(U + BG)
            return theta - thetaBG
        def spongeLayer(x):
            return 0

        def S_e(t,x,U,DU):
            BG, _, _, _, _ = Model.bg(t,x)

            U_tot = U + BG
            S_tot = Model.S_e(t,x, U_tot, DU)

            S_bg = Model.S_e(t,x, BG, DU)

            sponge = BGModel.spongeLayer(x)
            # return (S_tot - S_bg)*(1-sponge)
            return S_tot - S_bg - sponge*U

        if hasattr(Model, "F_c"):
            # overloaded flux applying bgFix
            def F_c(t,x,U):
                BG, _, _, _, _ = Model.bg(t,x)

                U_tot = U + BG
                res_tot = Model.F_c(t,x, U_tot)

                res_bg = Model.F_c(t,x, BG)

                sponge = BGModel.spongeLayer(x)
                # return (res_tot - res_bg)*(1-sponge) # + res_bg*sponge
                return res_tot - res_bg
            # interface method needed for LLF and time step control
            def maxWaveSpeed(t,x,U,n):
                BG, _, _, _, _ = Model.bg(t,x)
                U_tot = U + BG
                return Model.maxWaveSpeed(t,x,U_tot, n)

        if hasattr(Model, "F_v"):
            # overloaded viscous flux applying bgFix
            def F_v(t,x,U, DU):
                BG, _, _, _, _ = Model.bg(t,x)
                DBG = Model.jacBg(t,x)

                U_tot = U + BG
                DU_tot = DU + DBG

                res_tot = Model.F_v(t,x, U_tot, DU_tot)
                return res_tot
            # interface method needed for LLF and time step control
            def maxDiffusion(t,x,U):
                BG, _, _, _, _ = Model.bg(t,x)
                U_tot = U + BG
                return Model.maxDiffusion(t,x,U_tot)

        if hasattr(Model, "pressureTemperature"):
            pressureTemperature = Model.pressureTemperature

        def background(t,x):
            BG, _, _, _, _ = Model.bg(t,x)
            return BG

        def velocity(t,x,U):
            BG, _, _, _, _ = Model.bg(t,x)
            U_tot = U + BG
            return Model.velocity(t,x,U_tot)

        def physical(t,x,U):
            BG, _, _, _, _ = Model.bg(t,x)
            U_tot = U + BG
            return Model.physical(t,x,U_tot)

        def jump(t,x,U,V):
            BG, _, _, _, _ = Model.bg(t,x)
            U_tot = U + BG
            V_tot = V + BG
            return Model.jump(t,x, U_tot, V_tot)

        def bndFix(bnd,bndId):
            def dirichlet(t,x,U):
                BG, _, _, _, _ = Model.bg(t,x)
                U_tot = U + BG
                U_bnd  = bnd(t,x, U_tot)
                return U_bnd - BG
            def flux1(t,x,U,n):
                BG, _, _, _, _ = Model.bg(t,x)
                U_tot = U + BG
                U_bnd  = bnd(t,x, U_tot, n)
                BG_bnd = bnd(t,x, BG, n)
                return U_bnd - BG_bnd
            def flux2(t,x,U,n,k):
                BG, _, _, _, _ = Model.bg(t,x)
                U_tot = U + BG
                U_bnd  = bnd(t,x, U_tot, n, k)
                BG_bnd = bnd(t,x, BG, n, k)
                return U_bnd - BG_bnd

            # need to add fix for bnd being a tuple of diffusive and advective flux
            sig = signature( bnd )
            if( len(sig.parameters) == 3 ):
                parprint("exchanged bnd condition for",bndId,
                      "using dirichlet data")
                return dirichlet
            if( len(sig.parameters) == 4 ):
                parprint("exchanged bnd condition for",bndId,
                      "using fixed adv-flux data")
                return flux1
            elif ( len(sig.parameters) == 5 ):
                parprint("exchanged bnd condition for",bndId,
                      "using fixed diff-flux function")
                return flux2
            else:
                assert False, "Length of boundary data does not fit"

    BGModel.boundary = {}
    # convert original bnda data to bgFixed ones
    for bndId, bndFct in Model.boundary.items():
        if isinstance(bndFct,(tuple, list)):
            BGModel.boundary[ bndId ] = list()
            for fct in bndFct:
                BGModel.boundary[ bndId ].append( BGModel.bndFix(fct,bndId) )
        else:
            BGModel.boundary[ bndId ] = BGModel.bndFix(bndFct,bndId)

    # get BG and subtract from U0
    BG, _,_, theta,_  = Model.bg(0., x)

    BGModel.U0 = Model.U0 - BG

    return BGModel
