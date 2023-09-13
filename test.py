from icecream import ic
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize


def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0, nrepeat=5):
    # Randomly generated matrix that will be used to generate the eigenvectors
    QRmat = -1.0 + 2 * np.random.uniform(size=(n, n))

    Q, _ = np.linalg.qr(QRmat, mode="complete")  # Construct Q via a Q-R decomposition

    if nrepeat == 1:
        lam = np.random.uniform(low=eig_low, high=eig_high, size=n)
    else:
        lam = np.hstack(
            (
                eig_low * np.ones(nrepeat),
                np.random.uniform(low=eig_low, high=eig_high, size=n - nrepeat),
            )
        )

    # lam[1] = lam[0] + 1e-6

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute A = Q*Lambda*Q^{T}


def compute_lam_ab(lam, a, b):
    """
    Compute the lam_a and lam_b values
    """
    # lam_a = np.argmin(np.abs(lam - a)) - np.min(lam)
    # lam_b = np.argmin(np.abs(lam - b)) + np.min(lam)
    lam_a = np.min(lam[lam > a]) - np.min(lam)
    lam_b = np.max(lam[lam < b]) + np.min(lam)
    # lam_a = np.min(lam[lam > a])
    # lam_b = np.max(lam[lam < b])
    return lam_a, lam_b


def compute_eta(rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    eta = np.zeros(len(lam))
    for i in range(len(lam)):
        a = mp.tanh(rho * (mp.mpf(lam[i]) - mp.mpf(lam_a)))
        b = mp.tanh(rho * (mp.mpf(lam[i]) - mp.mpf(lam_b)))
        eta[i] = np.float64(a - b)

        if i == 0:
            ic(a, b, eta[i])

    return eta / mp.mpf(np.sum(eta))


def compute_h(eta, Q, D, n=None):
    """
    Compute the h value
    """
    if n is None:
        n = np.shape(Q)[0]
    
    h = 0.0
    for i in range(n):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def func2(D, A, B):
    Binv = np.linalg.inv(B)
    exp = expm(-rho * A)
    exp /= np.trace(exp)
    h = np.trace(D @ exp @ Binv)
    return h


def deriv(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    # Compute the eigenvalues of the generalized eigen problem
    lam, Q = eigh(A, B)

    lam_a, lam_b = compute_lam_ab(lam, a, b)

    eta = compute_eta(rho, lam, lam_a, lam_b)

    h = compute_h(eta, Q, D, np.shape(A)[0])

    hdot = np.zeros(ndvs)
    
    for i in range(np.shape(A)[0]):
        Ar = A - lam[i] * B
        Br = 2 * B @ Q[:, i]
        Mat = np.block([[Ar, Br], [-0.5 * Br.T, 0.0]])
        
        b0 = -2 * eta[i] * D @ Q[:, i]
        a = np.tanh(rho * (lam[i] - lam_a)) + np.tanh(rho * (lam[i] - lam_b))
        b1 = rho * eta[i] * (a * Q[:, i].T @ D @ Q[:, i] - h)
        b = np.block([[b0], [b1]])
        
        x = np.linalg.solve(Mat, b)
        
        pR_px = np.block([[(Adot - lam[i]*Bdot) @ Q[:, i]], [Q[:, i].T @ Bdot @ Q[:, i]]])
        
        hdot += x.T @ pR_px
        
        if Ddot is not None:
            hdot += eta[i] * Q[:, i].T @ Ddot @ Q[:, i]
        
        

def precise(rho, trace, lam_min, lam1, lam2):
    """
    Compute the precise value of the E_{ij} term

        E_{ij} = exp(-rho * (lam1 - lam_min)) / trace

        if lam1 == lam2:
            E_{ij} = exp(-rho * (lam1 - lam_min)) / trace
        else:
            E_{ij} = (exp(-rho * (lam1 - lam_min)) - exp(-rho * (lam2 - lam_min))) / (lam1 - lam2) / trace
    """

    with mp.workdps(80):
        if lam1 == lam2:
            val = -rho * mp.exp(-rho * (lam1 - lam_min)) / trace
        else:
            val = (
                (mp.exp(-rho * (lam1 - lam_min)) - mp.exp(-rho * (lam2 - lam_min)))
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


def precise2(rho, trace, lam1, lam2, lam_a, lam_b):
    """
    Compute the precise value of the E_{ij} term

        E_{ij} = tanh(rho * (lam1 - lam_min)) / trace

        if lam1 == lam2:
            E_{ij} = rho * tanh(rho * (lam1 - lam_min)) / trace
        else:
            E_{ij} = (tanh(rho * (lam1 - lam_min)) - tanh(rho * (lam2 - lam_min))) / (lam1 - lam2) / trace
    """
    with mp.workdps(80):
        a1 = mp.tanh(rho * (mp.mpf(lam1) - mp.mpf(lam_a)))
        b1 = mp.tanh(rho * (mp.mpf(lam1) - mp.mpf(lam_b)))
        a2 = mp.tanh(rho * (mp.mpf(lam2) - mp.mpf(lam_a)))
        b2 = mp.tanh(rho * (mp.mpf(lam2) - mp.mpf(lam_b)))

        if lam1 == lam2:
            val = rho * (a1 - b1) / mp.mpf(trace)
            ic("lam1 == lam2")
        else:
            val = (
                ((a1 - b1) - (a2 - b2)) / (mp.mpf(lam1) - mp.mpf(lam2)) / mp.mpf(trace)
            )
    return np.float64(val)


# Set parameters
rho = 1000.0
N = 10
n = 100
dh = 1e-30
ndvs = 1

np.random.seed(12345)

x = 0.1 * np.ones(ndvs)
p = np.random.uniform(size=ndvs)

A = rand_symm_mat(n)
B = rand_symm_mat(n)
# B = np.eye(n)
Adot = rand_symm_mat(n)
Bdot = rand_symm_mat(n)
Ddot = rand_symm_mat(n)
D = rand_symm_mat(n)

lam, Q = eigh(A, B)

lam, Q = eigh(A, B)
eta = np.exp(-rho * (lam - np.min(lam)))
eta = eta / np.sum(eta)

h1 = compute_h(eta, Q, D)
h2 = func2(D, A, B)

ic(h1)
ic(h2)

err_h = np.abs(h1 - h2) / np.abs(h1)
ic(err_h)

aa = expm(-rho * np.diag(lam))
bb = Q.T @ expm(-rho * A) @ Q
ic(np.allclose(aa, bb))

ic(np.allclose(A, Q @ Q.T @ A))
ic(np.allclose(A, A @ Q @ Q.T))

# ic(lam)
# lam = np.tanh(0.5*lam)
# ic(lam)

# set lam_a be the closest eigenvalue to 1.0 but greater than 1.0
a = 0.0
b = 20.0
lam_a, lam_b = compute_lam_ab(lam, a, b)

ic(lam_a, lam_b)
eta = np.tanh(rho * (lam - lam_a)) - np.tanh(rho * (lam - lam_b))
eta2 = np.tanh(rho * (lam - lam_a)) + np.tanh(rho * (lam - lam_b))
trace = np.sum(eta)
ic(trace)
# count the number of eigenvalues that are greater than lam_a and less than lam_b
count = np.sum((lam > lam_a) & (lam < lam_b))
ic(count * 2)
eta = eta / trace
trace2 = np.sum(eta2)
eta2 = eta2 / trace2
# ic(eta)
# ic(eta2)

ic(1 / count)
# eta = compute_eta(rho, lam, a, b)

ic(eta[0])
ic(eta[1])
ic((mp.mpf(eta[1]) - mp.mpf(eta[0])) / (mp.mpf(lam[1]) - mp.mpf(lam[0])))
ic(rho * eta[0])


E01 = precise2(rho, trace, lam[0], lam[1], lam_a, lam_b)
ic(E01)

h = compute_h(eta, Q, D, n)

# plot eta vs lam
plt.plot(lam, eta * trace, "o")
plt.plot(lam, eta2 * trace2, "o")
plt.show()
