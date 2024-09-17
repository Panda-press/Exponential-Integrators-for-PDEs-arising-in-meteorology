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

        self.op = op
        self.initial_condition = initial_condition
        self.seed_time = seed_time
        self.setup_stepper = setup_stepper(op, **stepper_args)
        self.folder = "FETests/Problem:{0}_Setup_Tau:{2}_Setup_Stepper:{1}_Seed_Time:{3}".format(problem_name,self.setup_stepper.name,setup_tau,self.seed_time)
        
        
        self.run_setup(setup_tau, self.setup_stepper, initial_condition)

    def get_initial_file(self):
        return self.folder + "_Initial.pickle"
    
    def get_target_results_file(self, tau, end_time):
        return self.folder + "_Tau:{0}_End_Time:{1}_Target.pickle".format(tau, end_time)

    def get_test_results_file(self, tau, stepper, end_time):
        return self.folder + "_Tau:{0}_Stepper:{1}_End_Time:{2}_Test.pickle".format(tau, stepper.name, end_time)

    def run_setup(self, tau, setup_stepper, initial_condition):
        # File path to initial data
        self.intitial_file_name = self.get_initial_file()

        # Check if file alread exist
        if os.path.isfile(self.intitial_file_name):
            with open(self.intitial_file_name, 'rb') as file:
                self.initial_condition.as_numpy[:] = pickle.load(file)

        # If not generate it
        else:
            time = Constant(0,"time")
            initial_condition = self.run(tau, setup_stepper, initial_condition, 0, self.seed_time)
            with open(self.intitial_file_name, 'wb') as file:
                pickle.dump(initial_condition.as_numpy[:], file)
    
    def run_test(self, tau, test_stepper, stepper_args, end_time):
        
        # Load initial conditions
        with open(self.intitial_file_name, 'rb') as file:
            self.initial_condition.as_numpy[:] = pickle.load(file)

        # Generate target data if it doesn't exist
        target_file_name = self.get_target_results_file(tau, end_time)
        if not os.path.isfile(target_file_name):
            self.target = self.run(tau, self.setup_stepper, self.initial_condition, self.seed_time, end_time)
            with open(target_file_name, 'wb') as file:
                pickle.dump(self.target.as_numpy[:], file)

        # Generate test stepper data if it doesn't exist
        test_stepper = test_stepper(op, **stepper_args)
        test_file_name = self.get_test_results_file(tau, test_stepper, end_time)
        if not os.path.isfile(test_file_name):
            test_results = self.run(tau, test_stepper, self.initial_condition, self.seed_time, end_time)
            with open(test_file_name, 'wb') as file:
                pickle.dump(test_results.as_numpy[:], file)
        

    def run(self, tau, stepper, initial_condition, start_time, end_time):
        # Runs for a given stepper
        current_step = initial_condition.copy()
        time = Constant(self.seed_time)
        while time.value < end_time:
            stepper(target=current_step, tau = tau)
            time.value += tau
        return current_step

    def produce_results(self, tau, stepper, stepper_args, end_time):
        self.run_test(tau, stepper, stepper_args, end_time)
        stepper = stepper(self.op, **stepper_args)
        test_file_name = self.get_test_results_file(tau, stepper, end_time)
        target_file_name = self.get_target_results_file(tau, end_time)
        test_results = self.initial_condition.copy()
        target_results = self.initial_condition.copy()
        with open(test_file_name, 'rb') as file:
            test_results.as_numpy[:] = pickle.load(file)
        with open(target_file_name, 'rb') as file:
            target_results.as_numpy[:] = pickle.load(file)
        test_results.plot()
        
if __name__ == "__main__":
    from allenCahn import dimR, time, sourceTime, domain
    from allenCahn import test2 as problem

    gridView = view(leafGridView(cartesianDomain(*domain)) )
    space = lagrange(gridView, order=1, dimRange=dimR)

    model, T, tauFE, u0, exact = problem(gridView)
    op = galerkin(model, domainSpace=space, rangeSpace=space)

    tau = 1e-2

    u_h = space.interpolate(u0, name='u_h')

    exp_arnoldi_stepper, args = steppersDict["EXPARN"]

    if "exp_v" in args.keys():
        m = 5
        args["expv_args"] = {"m":m}

    end_time = 3

    tester = Tester(u_h, op, "Allen Cahn", 1)

    
    tester.produce_results(tau, exp_arnoldi_stepper, args, end_time)

