# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pickle

e_thresholds = [1e-6, 1e-10]

# %%
data = pd.read_csv("Experiment_Data.csv")
ns = data["N"].unique()
# %%
for n in ns:
    ndata = data[data["N"] == n]
    for method in ndata["Method"].unique():        
        methoddata = ndata[ndata["Method"] == method]
        methoddata = methoddata.sort_values("M")

        plt.figure(1)
        if method == "Scipy":
            mean_Error = np.mean(methoddata["Error"])
        else:
            plt.loglog(methoddata["M"], methoddata["Error"], label= method)
        plt.xlabel("Dimention of Kyrlov subspace $m$")
        plt.ylabel("Error")
        plt.title("Graph Compairing M and Error for N={0}".format(n))
        plt.legend()

        plt.figure(2)
        if method == "Scipy":
            mean_computation_time = np.mean(methoddata["Computation Time"])
            plt.hlines(mean_computation_time, np.min(ndata["M"]), np.max(ndata["M"]), colors="r", linestyles="dashed", label="Scipy")
        else:
            plt.loglog(methoddata["M"], methoddata["Computation Time"], label= method)
        plt.xlabel("Dimention of Kyrlov subspace $m$")
        plt.ylabel("Computation Time $s$")
        plt.title("Graph Compairing M and Computation Time for N={0}".format(n))
        plt.legend()
        
        plt.figure(3)
        if method == "Scipy":
            plt.vlines(mean_computation_time, np.min(ndata["Error"]), np.max(ndata["Error"]), colors="r", linestyles="dashed", label="Scipy")
        else:
            plt.loglog(methoddata["Computation Time"], methoddata["Error"], label= method)
        plt.xlabel("Computation Time $s$")
        plt.title("Graph Compairing Computation Time and Error for N={0}".format(n))
        plt.ylabel("Error")
        plt.legend()
    
    plt.figure(1)
    plt.grid(True, which="both")
    plt.savefig("Plots/M v E Results for N={0}".format(n))
    plt.close()
    plt.figure(2)
    plt.grid(True, which="both")
    plt.savefig("Plots/M v Comp Time Results for N={0}".format(n)) 
    plt.close()
    plt.figure(3)
    plt.grid(True, which="both")
    plt.savefig("Plots/Compt Time v E Results for N={0}".format(n))
    plt.close()

# %% Plotting time for error to be bellow given bound
for e_threshold in e_thresholds:
    cut_data = data[data["Error"] < e_threshold]
    ns = cut_data["N"].unique()
    for method in data["Method"].unique():
        method_data = cut_data[cut_data["Method"] == method]
        result = []
        for n in ns:
            ndata = method_data[method_data["N"] == n]
            if method == "Scipy":
                continue
            try:
                result.append(ndata["Computation Time"].to_list()[ndata["Error"].argmin()])
            except:
                result.append(None)
        try:
            plt.loglog(ns, result, label = method)
        except:
            continue
    plt.legend()
    plt.title("A graph Showing the time to get below an error of {0} for different matrix sizes".format(e_threshold))
    plt.xlabel("N")
    plt.ylabel("Computation Time")
    plt.grid(True, which="both")
    plt.savefig("Plots/time to get below an error of {0}.png".format(e_threshold))
    plt.close()



# %%
