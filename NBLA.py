# From A RATIONAL KRYLOV SUBSPACE METHOD FOR THE COMPUTATION OF THE MATRIX EXPONENTIAL OPERATOR by H. BARKOUKI ∗, A. H. BENTBIB† , AND K. JBILOU ‡
import numpy as np
from numpy.linalg import norm
import scipy
from scipy.linalg import expm
from scipy.sparse import diags, csc_matrix, csr_matrix, lil_matrix, lil_array
import scipy.sparse
from scipy.sparse.linalg import expm, expm_multiply

def NBLA(A, B, C, m):
    #Has not been optimised
    H = lil_array((m+1, m+1))
    V = np.zeros((np.shape(B)[-1], m+2))
    W = np.zeros_like(V)
    delta, beta = 1, C.T@B
    returnbeta = beta
    V[:,0] = B/beta
    W[:,0] = C
    Vdash = A@V[:,0]
    Wdash = A.T@W[:,0]

    a = np.zeros((m))
    f=0
    for j in range(0, m):
        a = W[:,j].T @ Vdash 
        Vdash -= a * V[:,j]
        Wdash -= a * W[:,j]

        V[:,j+1], beta = Vdash/norm(Vdash), norm(Vdash)
        W[:,j+1], delta = Wdash/norm(Wdash), norm(Wdash)
        delta = delta.T

        Sigma = W[:,j+1].T@V[:,j+1]
        delta *= np.sqrt(Sigma)
        beta *= np.sqrt(Sigma) 

        V[:,j+1] *= np.sqrt(Sigma)
        W[:,j+1] *= np.sqrt(Sigma)

        Vdash = A @ V[:,j+1] - V[:,j] * delta
        Wdash = A.T @ W[:,j+1] - W[:,j] * beta.T
        H[j,j] = a
        H[j+1,j] = beta
        H[j,j+1] = delta

    return csc_matrix(H[0:m,0:m]), csc_matrix(V[:,0:m]), csc_matrix(W[:,0:m]), returnbeta

def NBLAExp(A, v_1, m):
    H, V, W, beta = NBLA(A, v_1, v_1, m)
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

    print(NBLAExp(A, v, m))
    print(expm_multiply(A, v))

