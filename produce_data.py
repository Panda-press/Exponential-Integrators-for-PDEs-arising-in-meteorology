import numpy as np
import pandas as pd
import time as time
import pickle as pickle
from numpy.linalg import norm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import diags, csc_matrix
import scipy.sparse
from Arnoldi import ArnoldiExp
from Stable_Lanzcos import LanzcosExp
from NBLA import NBLAExp
from kiops import KiopsExp

# Collects data on performance of different methods
# Independant Variables:
# $N$ size of the matrix
# #$M$ size of resultant matrix

# Error
# Computation time

def GetA(n):
    A = diags([-1,2,-1], [-1,0,1], shape=(n,n))
    A = csc_matrix(A)
    return A / n**2

methods = {
    "Scipy": lambda A, v, m: expm_multiply(A,v),
    "Arnoldi": ArnoldiExp,
    "Lanzcos": LanzcosExp#,
    #"Kiops": lambda A, V, m: KiopsExp(A,V)
    #"NBLA": NBLAExp
}
m = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
n = [2**i for i in range(0, 20)]
print(n)

v = np.random.rand(np.max(n))
method_names = list(methods.keys())
mlen = len(m)
nlen = len(n)
method_names_len = len(methods)
n_data = np.repeat(n, mlen * method_names_len)
m_data = np.tile(m, nlen * method_names_len)
method_names_data = np.tile(method_names, nlen * mlen)



data = pd.DataFrame(columns=["N","M","Method","Error","Computation Time"])

data["M"] = m_data
data["N"] = n_data
data["Method"] = method_names_data

errors = []
computation_times = []


true_values = []
scipy_computing_time =[]
for N in n:
    print("Computing {0}".format(N))
    A = GetA(N)
    V = v[0:N]
    start = time.time()
    result = methods["Scipy"](A, V, 1)
    end = time.time()
    true_values.append(result)
    scipy_computing_time.append(end-start)

for index, row in data.iterrows():

    if row["N"] < row["M"]:
        errors.append(None)
        computation_times.append(None)
    else:
        method = methods[row["Method"]]
        N = row["N"]
        M = row["M"]
        print("Method: {2}, N:{0} ,M:{1}".format(N, M, row["Method"]))
        if (row["Method"] == "Scipy"):
            errors.append(0.0)
            computation_times.append(scipy_computing_time[n.index(N)])
            continue

        A = GetA(N)
        V = v[0:N]

        # Gets an average
        total_time = 0
        total_error = 0
        count = 10
        for i in range(count):
            start = time.time()
            result = method(A, V, M)
            end = time.time()
            total_time += end - start
            total_error += norm(result - true_values[n.index(N)])
        average_time = total_time/count   
        average_error = total_error/count  
        
        computation_times.append(average_time)
        errors.append(average_error)

data["Error"] = errors
data["Computation Time"] = computation_times

data.dropna()

data.to_csv("Experiment_Data.csv")

