import numpy as np
import pickle as pickle
import os as os
from ufl import *
from dune.ufl import Space, Constant
from dune.fem.function import gridFunction
from dune.fem.space import lagrange
from dune.grid import cartesianDomain
from dune.fem.operator import galerkin
from dune.alugrid import aluCubeGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView as view
from steppers import BEStepper

class Tester():
    def __init__(self, 
                 initial_condition, 
                 op, 
                 problem_name, 
                 seed_time, 
                 end_time,
                 exact = None,
                 setup_tau = 1e-2,
                 setup_stepper = BEStepper,
                 **stepper_args):
        assert seed_time < end_time
        self.initial_condition = initial_condition
        self.op = op
        self.problem_name = problem_name
        self.seed_time = seed_time
        self.end_time = end_time
        if exact == None:
            self.run_setup(setup_tau, setup_stepper, stepper_args)

    def run_setup(self, tau, setup_stepper, stepper_args = {}):
        setup_stepper = setup_stepper(N = self.op, **stepper_args)
        file_name = "FETests/Problem:{0},Setup_Stepper:{1},Tau:{2}.pickle".format(self.problem_name,setup_stepper.name,tau)
        if os.path.isfile(file_name):
            with open(file_name, 'wb') as file:
                self.initial_condition = pickle.load(self.initial_condition, file)
        else:
            time = Constant(0,"time")
            while time.value < self.seed_time:
                print(time.value)
                setup_stepper(target=self.initial_condition, tau=tau)
                time.value += tau
            with open(file_name, 'wb') as file:
                pickle.dump(self.initial_condition, file)
    

        
if __name__ == "__main__":
    from allenCahn import dimR, time, sourceTime, domain
    from allenCahn import test2 as problem

    gridView = view(leafGridView(cartesianDomain(*domain)) )
    space = lagrange(gridView, order=1, dimRange=dimR)

    model, T, tauFE, u0, exact = problem(gridView)
    op = galerkin(model, domainSpace=space, rangeSpace=space)

    u_h = space.interpolate(u0, name='u_h')

    tester = Tester(u_h, op, "Allen Cahn", 1, 2)