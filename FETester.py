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
from steppers import BEStepper, steppersDict

class Tester():
    def __init__(self, 
                 initial_condition, 
                 op, 
                 problem_name, 
                 seed_time = 0, 
                 exact = None,
                 setup_tau = 1e-2,
                 setup_stepper = BEStepper,
                 **stepper_args):

        self.initial_condition = initial_condition
        self.op = op
        self.problem_name = problem_name
        self.seed_time = seed_time
        self.setup_stepper = setup_stepper(op, **stepper_args)
        self.folder = "FETests/Problem:{0}_Setup_Stepper:{1}_Setup_Tau:{2}_".format(self.problem_name,self.setup_stepper.name,setup_tau)
        
        
        self.run_setup(setup_tau, self.setup_stepper)

    def run_setup(self, tau, setup_stepper):
        intitial_file_name = self.folder + "Initial.pickle"
        if os.path.isfile(intitial_file_name):
            with open(intitial_file_name, 'rb') as file:
                self.initial_condition.as_numpy[:] = pickle.load(file)
        else:
            time = Constant(0,"time")
            while time.value < self.seed_time:
                self.setup_stepper(target=self.initial_condition, tau=tau)
                time.value += tau
            with open(intitial_file_name, 'wb') as file:
                pickle.dump(self.initial_condition.as_numpy[:], file)
    
    def run_test(self, tau, end_time, test_stepper, stepper_args):
        current_step = self.initial_condition.copy()

        # Check or generate target values
        self.target_file_name = self.folder + "Target_Tau:{0}.pickle".format(tau)
        if not os.path.isfile(self.target_file_name):
            self.target = self.run(tau, self.setup_stepper, end_time)
            with open(self.target_file_name, 'wb') as file:
                pickle.dump(self.target.as_numpy[:], file)

        # Check or generate data for test stepper
        test_stepper = test_stepper(op, **stepper_args)
        test_file_name = self.folder + "Test_Tau:{0}_Method:{1}.pickle".format(tau,test_stepper.name)
        if not os.path.isfile(test_file_name):
            test_results = self.run(tau, test_stepper, end_time)
            with open(test_file_name, 'wb') as file:
                pickle.dump(test_results.as_numpy[:], file)
        

    def run(self, tau, stepper, end_time):
        current_step = self.initial_condition.copy()
        time = Constant(self.seed_time)
        while time.value < end_time:
            stepper(target=current_step, tau = tau)
            time.value += tau
        return current_step


        
if __name__ == "__main__":
    from allenCahn import dimR, time, sourceTime, domain
    from allenCahn import test2 as problem

    gridView = view(leafGridView(cartesianDomain(*domain)) )
    space = lagrange(gridView, order=1, dimRange=dimR)

    model, T, tauFE, u0, exact = problem(gridView)
    op = galerkin(model, domainSpace=space, rangeSpace=space)

    u_h = space.interpolate(u0, name='u_h')

    tester = Tester(u_h, op, "Allen Cahn", 1)

    exp_arnoldi_stepper, args = steppersDict["EXPARN"]

    if "exp_v" in args.keys():
        m = 5
        args["expv_args"] = {"m":m}
    tester.run_test(1e-2, 2, exp_arnoldi_stepper, args)

