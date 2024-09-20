import numpy as np
import pickle as pickle
import os as os
import hashlib
import copy
import pandas as pd
import matplotlib.pyplot as plt
from ufl import *
from dune.ufl import Space, Constant
from dune.fem import integrate
from dune.fem.function import gridFunction
from dune.fem.space import lagrange
from dune.grid import cartesianDomain
from dune.fem.operator import galerkin
from dune.alugrid import aluCubeGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView as view
from steppers import BEStepper, steppersDict

# results_folder = "FETests"
results_folder = "FETests2009"

class Tester():
    def __init__(self, 
                 initial_condition, 
                 op, 
                 problem_name, 
                 seed_time = 0, 
                 setup_tau = 1e-2,
                 setup_stepper = BEStepper,
                 exact = None,
                 **stepper_args):

        self.op = op
        self.N = self.op.domainSpace.gridView.size(0)
        self.initial_condition = initial_condition
        self.seed_time = seed_time
        self.setup_stepper = setup_stepper(op, **stepper_args)
        self.folder = results_folder + "/Problem:{0}_Grid_Size:{4}_Setup_Tau:{2}_Setup_Stepper:{1}_Seed_Time:{3}".format(problem_name,self.setup_stepper.name,setup_tau,self.seed_time,self.N)
        self.exact = exact
        
        self.run_setup(setup_tau, initial_condition)

    def get_initial_file(self):
        return self.folder + "_Initial.pickle"
    
    def get_target_results_file(self, tau, end_time):
        return self.folder + "_Tau:{0}_End_Time:{1}_Target.pickle".format(tau, end_time)

    def get_test_results_file(self, tau, stepper_name, end_time):
        return self.folder + "_Tau:{0}_End_Time:{1}_Stepper:{2}.pickle".format(tau, end_time, stepper_name)

    def run_setup(self, tau, initial_condition):
        if self.exact is not None:
            return

        # File path to initial data
        self.intitial_file_name = self.get_initial_file()

        # Check if file alread exist
        if os.path.isfile(self.intitial_file_name):
            with open(self.intitial_file_name, 'rb') as file:
                self.initial_condition.as_numpy[:] = pickle.load(file)

        # If not generate it
        else:
            self.initial_condition, _ = self.run(tau, self.setup_stepper, initial_condition, 0, self.seed_time)
            with open(self.intitial_file_name, 'wb') as file:
                pickle.dump(self.initial_condition.as_numpy[:], file)
    
    def run_test(self, tau, test_stepper, stepper_args, end_time):
        
        # Load initial conditions
        if self.exact is not None:
            self.initial_condition.interpolate(self.exact(self.seed_time))
        else:
            with open(self.intitial_file_name, 'rb') as file:
                self.initial_condition.as_numpy[:] = pickle.load(file)

        # Generate target data if it doesn't exist
        if self.exact is None:
            target_file_name = self.get_target_results_file(tau, end_time)
            if not os.path.isfile(target_file_name):
                self.target,self.target_countN = self.run(tau, self.setup_stepper, self.initial_condition, self.seed_time, end_time)
                with open(target_file_name, 'wb') as file:
                    pickle.dump([self.target.as_numpy[:],self.target_countN], file)
            else:
                self.target = self.initial_condition.copy()
                with open(target_file_name, 'rb') as file:
                    self.target.as_numpy[:], self.target_countN = pickle.load(file)
        else:
            self.target = gridFunction(exact(end_time),
                               gridView=op.domainSpace.gridView,order=5)
            self.target_countN = 0

        # Generate test stepper data if it doesn't exist
        test_stepper = test_stepper(op, **stepper_args)
        temp = copy.deepcopy(stepper_args)
        temp['exp_v'][0] = 0
        test_stepper_name = test_stepper.name +\
                            hashlib.sha1( repr(sorted(temp.items())).encode('utf-8') ).hexdigest()
        test_file_name = self.get_test_results_file(tau, test_stepper_name, end_time)
        if not os.path.isfile(test_file_name):
            self.test_results, self.test_countN = self.run(tau, test_stepper, self.initial_condition, self.seed_time, end_time)
            with open(test_file_name, 'wb') as file:
                pickle.dump([self.test_results.as_numpy[:],self.test_countN], file)
        else:
            self.test_results = self.initial_condition.copy()
            with open(test_file_name, 'rb') as file:
                self.test_results.as_numpy[:], self.test_countN = pickle.load(file)
        

    def run(self, tau, stepper, initial_condition, start_time, end_time):
        # Runs for a given stepper
        current_step = initial_condition.copy()
        time = start_time
        while time < end_time:
            stepper.N.model.sourceTime = time
            stepper(target=current_step, tau = tau)
            time += tau
        countN = stepper.countN
        stepper.countN = 0
        return current_step, countN

    def produce_results(self, tau, stepper, stepper_args, end_time):
        self.run_test(tau, stepper, stepper_args, end_time)

if __name__ == "__main__":
    if True:
        from allenCahn import dimR, time, sourceTime, domain
        from allenCahn import test2 as problem
        problemName = "Allen Cahn Test2"
        start_time = 1
        end_time = 3
    else:
        from parabolicTest import dimR, time, sourceTime, domain
        from parabolicTest import paraTest1 as problem
        problemName = "Parabolic Test1"
        tau0 = 1e-4  # 2e-3,N:32,Tau:0.002: compare m=5->m=10
        start_time = 0
        end_time = 0.001

    results = []


    exp_methods = ["EXPARN", "EXPLAN", "EXPKIOPS"]
    # exp_methods = ["EXPLAN", "EXPARN", "EXPKIOPS"]

    domain = list(domain)
    for exp_method in exp_methods:
        print("EXP method:{0}".format(exp_method))
        for N in [2**i for i in range(5, 8)]:
            print("N:{0}".format(N))
            for tau in tau0*np.array([1/2**i for i in range(0, 4)]):
                print("Tau:{0}".format(tau))

                domain[2] = [N,N]

                gridView = view(leafGridView(cartesianDomain(*domain)) )
                space = lagrange(gridView, order=1, dimRange=dimR)

                model, T, tauFE, u0, exact = problem(gridView)
                op = galerkin(model, domainSpace=space, rangeSpace=space)

                u_h = space.interpolate(u0, name='u_h')

                exp_stepper, args = steppersDict[exp_method]
                if "exp_v" in args.keys():
                    m = 5
                    args["expv_args"] = {"m":m}
                # else if kiops....

                tester = Tester(u_h, op, problemName, start_time, exact=exact)
                
                tester.produce_results(tau, exp_stepper, args, end_time)

                tester.test_results.plot()
                tester.target.plot()
                error = tester.test_results - tester.target
                ref = [ np.sqrt(r) for r in
                        integrate([tester.target**2,
                                   inner(grad(tester.target),grad(tester.target))]
                                 )]
                H1err = [ np.sqrt(e)/r
                          for r,e in zip(ref,integrate([error**2,inner(grad(error),grad(error))])) ]

                print(f"{exp_method},{tau},{gridView.size(0)}: {H1err}")
                # if exact is not None:
                #     exact_error = ...

                # write file self.test_results.plot()
                results += [ [exp_method,gridView.size(0),tau,H1err[0],H1err[1],
                                    tester.target_countN,tester.test_countN] ]

    # produce plots using 'results'

    results = pd.DataFrame(results)
    results.columns = ["Method", "Grid size", "Tau", "Error L2", "Error H1", "Target N Count", "Test N Count"]


    plt.figure(1)
    plt.scatter(results["Tau"], results["Target N Count"], marker=".", label="BE Method")
    for exp_method in exp_methods:
        trimmed_data = results[results["Method"] == exp_method]
        plt.figure(1)
        plt.scatter(trimmed_data["Tau"], trimmed_data["Test N Count"], marker=".", label=exp_method)
        plt.figure(2)
        plt.scatter(trimmed_data["Tau"], trimmed_data["Error L2"], marker=".", label=exp_method)
        print(trimmed_data["Tau"],trimmed_data["Error L2"])

    plt.figure(1)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.figure(2)
    plt.legend()
    plt.xscale('log')
    plt.show()
