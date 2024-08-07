# https://arc.aiaa.org/doi/epdf/10.2514/3.5878
import numpy as np
from numpy.linalg import norm
import scipy as scipy
from scipy.linalg import expm
from scipy.sparse import diags, csc_matrix, csr_matrix, lil_matrix, lil_array
import scipy.sparse
from scipy.sparse.linalg import expm, expm_multiply

def Lanzcos(A, _V_hat, m):
    n = np.shape(A)[-1]
    alpha = np.zeros((m+1))
    beta = np.zeros((m+1))
    v = np.zeros((n, m+1))
    v_hat = np.zeros((n, m+2))
    v_hat[:, 1] = _V_hat

    for i in range(1, m+1):
        beta[i] = norm(v_hat[:,i])
        v[:, i] = v_hat[:, i]/beta[i]
        alpha[i] = v[:, i].T @ A @ v[:, i]
        v_hat[:, i+1] = A@v[:,i] - alpha[i]*v[:,i] - beta[i]*v[:,i-1]
        

    beta = beta[2:m+1]
    alpha = alpha[1:m+1]

    return diags([beta, alpha, beta], [-1,0,1]), v[:, 1:m+1], norm(_V_hat)

def LanzcosExp(A, v_1, m):
    H, V, beta = Lanzcos(A, v_1, m)
    e_1 = np.zeros((m))
    e_1[0] = 1
    return V @ expm_multiply(H, e_1) * beta

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    n = 10
    m = 6

    v = np.random.rand(n)

    e_1 = np.zeros((m))
    e_1[0] = 1

    A = diags([-1,2,-1], [-1,0,1], shape=(n,n))
    A = csc_matrix(A)

    print(LanzcosExp(A, v, m))
    print(expm_multiply(A, v))

