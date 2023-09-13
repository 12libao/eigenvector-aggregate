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


def Mat_A(n, x):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = (1 + abs(i - j)) * x
    for i in range(n):
        A[i, i] = 3.0 * x - 2.0 * i
    return A


def Mat_dAdx(n):
    Ax = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                Ax[i, j] = 1 + abs(i - j)
    for i in range(n):
        Ax[i, i] = 3.0
    return Ax


def Mat_B(n, x):
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                B[i, j] = (20.0 - abs(i - j)) - x
    for i in range(n):
        B[i, i] = 3.0 * x + 2.0 * i
    return 0.1 * B + n * np.eye(n)  # make sure B is positive definite


def Mat_dBdx(n):
    Bx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                Bx[i, j] = -1
    for i in range(n):
        Bx[i, i] = 3
    return 0.1 * Bx


def Mat_D(n, x):
    D = np.eye(n) * x - np.ones((n, n)) * x * 0.1
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] += abs(i - j)
    return D


def Mat_dDdx(n):
    Dx = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                Dx[i, j] = -0.1
    for i in range(n):
        Dx[i, i] += 1.0 - 0.1
    return Dx


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
    N_b = n - np.sum(lam > lam_b)
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

    return eta / np.sum(eta)


def compute_eta_ab(lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    count = np.sum((lam > lam_a) & (lam < lam_b))
    eta = 1 / count
    return eta


def compute_h(eta, Q, D, n):
    """
    Compute the h value
    """
    h = 0.0
    for i in range(n):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def compute_approx_h(eta_ab, N_a, N_b, Q, D):
    """
    Compute the h value
    """
    h = 0.0
    for i in range(N_a, N_b):
        h += eta_ab * Q[:, i].T @ D @ Q[:, i]

    return h


def deriv(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    # Compute the eigenvalues of the generalized eigen problem
    lam, Q = eigh(A, B)

    lam_a, lam_b, _, _ = compute_lam_ab(lam, a, b)

    eta = compute_eta(rho, lam, lam_a, lam_b)

    h = compute_h(eta, Q, D, np.shape(A)[0])

    hdot = np.zeros(ndvs)

    for i in range(np.shape(A)[0]):
        Ar = A - lam[i] * B
        Br = (2 * B @ Q[:, i]).reshape(-1, 1)
        Mat = np.block([[Ar, Br], [-0.5 * Br.T, 0.0]])

        b0 = (-2 * eta[i] * D @ Q[:, i]).reshape(-1, 1)
        a = np.tanh(rho * (lam[i] - lam_a)) + np.tanh(rho * (lam[i] - lam_b))
        b1 = rho * eta[i] * (a * Q[:, i].T @ D @ Q[:, i] - h)
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


def deriv_approx(rho, A, B, D, Adot, Bdot, a, b, Ddot=None, ndvs=1):
    lam, Q = eigh(A, B)

    lam_a, lam_b, N_a, N_b = compute_lam_ab(lam, a, b)

    eta_ab = compute_eta_ab(lam, lam_a, lam_b)

    h = compute_approx_h(eta_ab, N_a, N_b, Q, D)

    C = B @ Q[:, N_a:N_b]
    U, _ = np.linalg.qr(C)
    Z = np.eye(np.shape(A)[0]) - U @ U.T

    hdot = np.zeros(ndvs)

    for i in range(N_a, N_b):
        for j in range(N_a, N_b):
            qDq = Q[:, i].T @ D @ Q[:, j]
            qBdotq = Q[:, i].T @ Bdot @ Q[:, j]
            hdot -= eta_ab * qDq * qBdotq

    for i in range(N_a, N_b):
        Dq = D @ Q[:, i]
        Ak = A - lam[i] * B
        Abar = Z.T @ Ak @ Z
        bbar = Z.T @ (-2.0 * Dq)
        phi = Z @ np.linalg.solve(Abar, bbar)

        hdot += (
            eta_ab * (phi.T + rho *  h * Q[:, i].T) @ (Adot - lam[i] * Bdot) @ Q[:, i]
        )

        if Ddot is not None:
            hdot += eta_ab * Q[:, i].T @ Ddot @ Q[:, i]

    return hdot


n = 10
eps = 1e-8
rho = 1000.0
ndvs = 5 

p = np.random.uniform(size=ndvs)

a = -1.0
b = 1.0

x = 10.0

A = Mat_A(n, x)
B = Mat_B(n, x)
D = Mat_D(n, x)

Ax = Mat_dAdx(n)
Bx = Mat_dBdx(n)
Dx = Mat_dDdx(n)

A1 = Mat_A(n, x + eps)
B1 = Mat_B(n, x + eps)
D1 = Mat_D(n, x + eps)

# check if: A = A^T
ic(np.allclose(A, A.T))
# check if: B = B^T and B is positive definite
ic(np.allclose(B, B.T) and np.all(np.linalg.eigvals(B) > 0))
# check if: D = D^T
ic(np.allclose(D, D.T))

# # check if: A Q = B Q Λ
# ic(np.allclose(A @ Q, B @ Q @ Λ))
# # check if: Q^T B Q = I
# ic(np.allclose(Q.T @ B @ Q, np.eye(n)))
# # check if: Q^T A Q = Λ
# ic(np.allclose(Q.T @ A @ Q, Λ))
# # check if: Q Q^T = B^{-1}
# ic(np.allclose(Binv, np.linalg.inv(B)))
# # check if: sum(η) = 1
# ic(np.allclose(np.sum(η), 1))

# check if: dAdx is correct
ic(np.allclose((A1 - A) / eps, Ax))
# check if: dBdx is correct
ic(np.allclose((B1 - B) / eps, Bx))
# check if: dDdx is correct
ic(np.allclose((D1 - D) / eps, Dx))

lam, Q = eigh(A, B)

ic(Q.T @ Q)
ic(np.allclose(Q.T @ B @ Q, np.eye(n)))
ic(np.allclose(Q @ B @ Q.T, np.eye(n)))
ic(np.allclose(Q @ Q.T, np.linalg.inv(B)))

(
    lam_a,
    lam_b,
    N_a,
    N_b,
) = compute_lam_ab(lam, a, b)
eta = compute_eta(rho, lam, lam_a, lam_b)
eta_ab = compute_eta_ab(lam, lam_a, lam_b)
h1 = compute_h(eta, Q, D, n)
h1_approx = compute_approx_h(eta_ab, N_a, N_b, Q, D)

ic(h1, h1_approx)

ic(lam)
ic(eta)
ic(compute_eta_ab(lam, lam_a, lam_b))

ic(N_a, N_b)

lam, Q = eigh(A1, B1)
eta = compute_eta(rho, lam, lam_a, lam_b)
h2 = compute_h(eta, Q, D1, n)

dh_fd = (h2 - h1) / eps
ic(dh_fd)
dh = deriv(rho, A, B, D, Ax, Bx, a, b, Ddot=Dx, ndvs=ndvs) @ p
ic(dh)
dh_approx = deriv_approx(rho, A, B, D, Ax, Bx, a, b, Ddot=Dx, ndvs=ndvs) @ p
ic(dh_approx)
error = np.abs(dh - dh_fd) / np.abs(dh)
error_approx = np.abs(dh_approx - dh) / np.abs(dh)
ic(error)
ic(error_approx)
