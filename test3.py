from icecream import ic
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pylab as plt
import mpmath as mp
import numpy as np
import scipy
from scipy import sparse
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve


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

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute A = Q*Lambda*Q^{T}


def precise(rho, trace, lam1, lam2, lam_a, lam_b):
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

        eta1 = a1 - b1
        eta2 = a2 - b2
        
        # ic(eta1, eta2)

        if lam1 == lam2:
            # val = -rho * eta1 * (a1 + b1)/ mp.mpf(trace)
            val = 0.0
        else:
            val = (eta1 - eta2) / (mp.mpf(lam1) - mp.mpf(lam2)) / mp.mpf(trace)
    return np.float64(val)


def preciseG(rho, trace, lam1, lam2, lam_a, lam_b):
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

        eta1 = a1 - b1
        eta2 = a2 - b2

        if lam1 == lam2:
            # val = -rho * lam1 * eta1 * (a1 + b1) / mp.mpf(trace)
            val = 0.0
        else:
            val = (
                (lam1 * eta1 - lam2 * eta2)
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


def compute_lam_ab(lam, a, b):
    """
    Compute the lam_a and lam_b values
    """
    # lam_a = np.argmin(np.abs(lam - a)) - np.min(lam)
    # lam_b = np.argmin(np.abs(lam - b)) + np.min(lam)
    lam_a = np.min(lam[lam > a]) - np.min(np.abs(lam))
    lam_b = np.max(lam[lam < b]) + np.min(np.abs(lam))
    # lam_a = np.min(lam[lam > a])
    # lam_b = np.max(lam[lam < b])
    N_a = np.sum(lam < lam_a)
    N_b = lam.shape[0] - np.sum(lam > lam_b)
    return lam_a, lam_b, N_a, N_b


def compute_eta(rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    eta = np.zeros(len(lam))
    for i in range(len(lam)):
        a = np.tanh(rho * (lam[i] - lam_a))
        b = np.tanh(rho * (lam[i] - lam_b))
        eta[i] = a - b

    return eta


def compute_eta2(rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    eta = np.zeros(len(lam))
    for i in range(len(lam)):
        a = np.tanh(rho * (lam[i] - lam_a))
        b = np.tanh(rho * (lam[i] - lam_b))
        eta[i] = a + b

    return eta


def compute_eta_ab(lam, N_a, N_b):
    """
    Compute the eta values
    """
    # count = np.sum((lam > lam_a) & (lam < lam_b))
    eta = 1 / (N_b - N_a) * np.ones(len(lam))
    return eta


def compute_h(eta, Q, D, n):
    """
    Compute the h value
    """
    h = 0.0
    for i in range(n):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def compute_approx_h(eta, N_a, N_b, Q, D):
    """
    Compute the h value
    """
    h = 0.0
    for i in range(N_a, N_b):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def deriv2(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    # Compute the eigenvalues of the generalized eigen problem
    lam, Q = eigh(A, B)

    lam_a, lam_b, _, _ = compute_lam_ab(lam, a, b)

    eta = compute_eta(rho, lam, lam_a, lam_b)
    trace = np.sum(eta)
    eta = eta / trace

    h = compute_h(eta, Q, D, np.shape(A)[0])

    hdot = np.zeros(ndvs)

    for i in range(np.shape(A)[0]):
        Ar = A - lam[i] * B
        Br = (2 * B @ Q[:, i]).reshape(-1, 1)
        Mat = np.block([[Ar, Br], [-0.5 * Br.T, 0.0]])

        b0 = (-2 * eta[i] * D @ Q[:, i]).reshape(-1, 1)
        a = np.tanh(rho * (lam[i] - lam_a)) + np.tanh(rho * (lam[i] - lam_b))
        b1 = rho * eta[i] * a * ( Q[:, i].T @ D @ Q[:, i] - h)
        b = np.block([[b0], [b1]])

        x = np.linalg.solve(Mat, b)

        pR_px = np.block(
            [
                [((Adot - lam[i] * Bdot) @ Q[:, i]).reshape(-1, 1)],
                [Q[:, i].T @ Bdot @ Q[:, i]],
            ]
        )

        hdot += np.dot(x.T, pR_px).reshape(-1)

        if Ddot is not None:
            hdot += eta[i] * Q[:, i].T @ Ddot @ Q[:, i]

    return hdot


def deriv(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    """
    Compute the forward mode derivative
    """

    # Compute the eigenvalues of the generalized eigen problem
    lam, Q = eigh(A, B)
    lam_a, lam_b, _, _ = compute_lam_ab(lam, a, b)
    eta = compute_eta(rho, lam, lam_a, lam_b)
    trace = np.sum(eta)
    eta = eta / trace

    h = compute_h(eta, Q, D, np.shape(A)[0])

    hdot = np.zeros(ndvs)

    for j in range(np.shape(A)[0]):
        for i in range(np.shape(A)[0]):
            Adot_q = Q[:, i].T @ Adot @ Q[:, j]
            Bdot_q = Q[:, i].T @ Bdot @ Q[:, j]
            qDq = Q[:, i].T @ D @ Q[:, j]

            Eij = precise(rho, trace, lam[i], lam[j], lam_a, lam_b)
            Gij = preciseG(rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                scale = h
            else:
                scale = qDq

            hdot += scale * (Eij * Adot_q - Gij * Bdot_q)

    for j in range(np.shape(A)[0]):
        hdot -= eta[j] * (Q[:, j].T @ D @ Q[:, j]) * (Q[:, j].T @ Bdot @ Q[:, j])

        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


def deriv_approx(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    lam, Q = eigh(A, B)

    lam_a, lam_b, N_a, N_b = compute_lam_ab(lam, a, b)

    # eta_ab = compute_eta_ab(lam, lam_a, lam_b)
    # h = compute_approx_h(eta_ab, N_a, N_b, Q, D)

    eta = compute_eta(rho, lam, lam_a, lam_b)
    trace = np.sum(eta)
    eta = eta / trace

    h = compute_h(eta, Q, D, np.shape(A)[0])

    C = B @ Q[:, N_a:N_b]
    U, _ = np.linalg.qr(C)
    Z = np.eye(np.shape(A)[0]) - U @ U.T

    hdot = np.zeros(ndvs)

    for i in range(N_a, N_b):
        for j in range(N_a, N_b):
            if i != j:
                qDq = Q[:, i].T @ D @ Q[:, j]
                qBdotq = Q[:, i].T @ Bdot @ Q[:, j]
                hdot -= eta[i] * qDq * qBdotq

    for i in range(N_a, N_b):
        Dq = D @ Q[:, i]
        Ak = A - lam[i] * B
        Abar = Z.T @ Ak @ Z
        bbar = Z.T @ (-2.0 * eta[i] * Dq)
        phi = Z @ np.linalg.solve(Abar, bbar)

        # hdot += eta[i] * rho * h * Q[:, i].T @ (Adot - lam[i] * Bdot) @ Q[:, i]

        hdot += Q[:, i].T @ (Adot - lam[i] * Bdot) @ phi

        if Ddot is not None:
            hdot += eta[i] * Q[:, i].T @ Ddot @ Q[:, i]

    return hdot


def deriv_approx2(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    lam, Q = eigh(A, B)
    
    # ic(np.dot(Q.T, Q))
    # ic(np.dot(Q, Q.T))
    

    lam_a, lam_b, N_a, N_b = compute_lam_ab(lam, a, b)

    eta_ab = compute_eta_ab(lam, N_a, N_b)
    ic(eta_ab[0], eta_ab[-1])
    h = compute_approx_h(eta_ab, N_a, N_b, Q, D)

    # eta = compute_eta(rho, lam, lam_a, lam_b)
    # trace = np.sum(eta)
    # eta = eta / trace

    # h = compute_h(eta, Q, D, np.shape(A)[0])

    C = B @ Q[:, N_a:N_b]
    U, _ = np.linalg.qr(C)
    Z = np.eye(np.shape(A)[0]) - U @ U.T

    hdot = np.zeros(ndvs)

    for i in range(N_a, N_b):
        for j in range(N_a, N_b):
            qDq = Q[:, i].T @ D @ Q[:, j]
            qAdotq = Q[:, i].T @ Adot @ Q[:, j]
            qBdotq = Q[:, i].T @ Bdot @ Q[:, j]

            Eij = precise(rho, trace, lam[i], lam[j], lam_a, lam_b)
            Gij = preciseG(rho, trace, lam[i], lam[j], lam_a, lam_b)
            
            if i == j:
                scale = qDq - h
            else:
                scale = qDq

            hdot += scale * (Eij * qAdotq - Gij * qBdotq)

    for i in range(N_a, N_b):
        Dq = D @ Q[:, i]
        Ak = A - lam[i] * B
        Abar = Z.T @ Ak @ Z
        
        # # print the condition number of Mat
        # ic(np.linalg.cond(Ak))
        # # compute the rank of Mat
        # ic(np.linalg.matrix_rank(Ak))
        
        # # print the condition number of Mat
        # ic(np.linalg.cond(Abar))
        # # compute the rank of Mat
        # ic(np.linalg.matrix_rank(Abar))
        
        bbar = Z.T @ (-2.0 * eta[i] * Dq)
        phi = Z @ np.linalg.solve(Abar, bbar)

        hdot -= eta[i] * (Q[:, i].T @ D @ Q[:, i]) * (Q[:, i].T @ Bdot @ Q[:, i])
        
        hdot += Q[:, i].T @ (Adot - lam[i] * Bdot) @ phi

        if Ddot is not None:
            hdot += eta[i] * Q[:, i].T @ Ddot @ Q[:, i]

    return hdot


# Set parameters
rho = 10.0
N = 10
n = 100
dh = 1e-30
ndvs = 5

np.random.seed(123)

x = 0.1 * np.ones(ndvs)
p = np.random.uniform(size=ndvs)

A = rand_symm_mat(n)
B = rand_symm_mat(n)
B = np.eye(n)
Adot = rand_symm_mat(n)
Bdot = rand_symm_mat(n)
Ddot = rand_symm_mat(n)
D = rand_symm_mat(n)

lam, Q = eigh(A, B)

# check the rank of Q
ic(np.linalg.matrix_rank(Q))

a = lam[0] + 10.0
b = lam[0] + 30.0

lam, Q = eigh(A, B)
lam_a, lam_b, _, _ = compute_lam_ab(lam, a, b)
eta = compute_eta(rho, lam, lam_a, lam_b)
eta2 = compute_eta2(rho, lam, lam_a, lam_b)

plt.plot(lam, eta, "o-")
plt.plot(lam, eta2, "o-")
plt.show()

# use forward difference to compute the derivative
lam, Q = eigh(A, B)
lam_a, lam_b, N_a, N_b = compute_lam_ab(lam, a, b)
eta = compute_eta(rho, lam, lam_a, lam_b)
trace = np.sum(eta)
eta = eta / trace

h = compute_h(eta, Q, D, np.shape(A)[0])

ans = np.dot(deriv(rho, A, B, D, Adot, Bdot, a, b, Ddot, ndvs=ndvs), p)
ans_approx0 = np.dot(deriv2(rho, A, B, D, Adot, Bdot, a, b, Ddot, ndvs=ndvs), p)
ans_approx = np.dot(deriv_approx(rho, A, B, D, Adot, Bdot, a, b, Ddot, ndvs=ndvs), p)
ans_approx2 = np.dot(deriv_approx2(rho, A, B, D, Adot, Bdot, a, b, Ddot, ndvs=ndvs), p)

print("ans = ", ans)
print("ans_approx0 = ", ans_approx0)
print("ans_approx = ", ans_approx)
print("ans_approx2 = ", ans_approx2)
print("error = ", np.abs(ans - ans_approx0) / np.abs(ans))
print("error = ", np.abs(ans - ans_approx) / np.abs(ans))
print("error = ", np.abs(ans - ans_approx2) / np.abs(ans))
