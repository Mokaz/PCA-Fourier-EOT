import numpy as np
from numpy.linalg import norm
import random

def angleBetweenVectors(p1, p2):
    q1 = p1/norm(p1)
    q2 = p2/norm(p2)
    return np.arccos(float(q1.T @ q2))

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return angle, r

def pol2cart(angle, r):
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return x, y

def vectorAngle(v):
    return float(np.arctan2(v[1], v[0]))

def dVectorAngle(v):
    vNormSquared = norm(v)**2
    return np.array([-v[1], v[0]]).reshape((1, 2))/vNormSquared

def ssa(angle):
    # Smallest Signed Angle
    return np.pi - np.mod(np.pi - angle, 2 * np.pi)

def rot2D(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])

def rotz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0, 0, 1]])

def roty(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle),  0, np.cos(angle)]])

def rotx(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])

def ur(angle):
    return np.array([[np.cos(angle)], [np.sin(angle)]])

def ut(angle):
    return np.array([[-np.sin(angle)], [np.cos(angle)]])

def Ur(angle):
    u = ur(angle)
    return u @ u.T

def cross2D(u, v):
    return u[0]*v[1] - u[1]*v[0]

def rot3DfromEulerAngles(eulerAngles):
    # eulerAngles = [yaw, pitch, roll]
    yaw = float(eulerAngles[0])
    pitch = float(eulerAngles[1])
    roll = float(eulerAngles[2])
    Rz = rotz(yaw)
    Ry = roty(pitch)
    Rx = rotx(roll)
    return Rz @ Ry @ Rx

def rot3DfromQuaternion(q):
    qq = q/norm(q)
    n = qq[0]
    e1 = qq[1]
    e2 = qq[2]
    e3 = qq[3]
    r11 = n**2 + e1**2 - e2**2 - e3**2
    r22 = n**2 - e1**2 + e2**2 - e3**2
    r33 = n**2 - e1**2 - e2**2 + e3**2
    r12 = 2*(e1*e2 - n*e3)
    r21 = 2*(e1*e2 + n*e3)
    r13 = 2*(e1*e3 + n*e2)
    r31 = 2*(e1*e3 - n*e2)
    r23 = 2*(e2*e3 - n*e1)
    r32 = 2*(e2*e3 + n*e1)
    return np.array([[r11, r12, r13],
                     [r21, r22, r23],
                     [r31, r32, r33]])

def eulerAnglesFromRot(R):
    r11 = float(R[0, 0])
    r21 = float(R[1, 0])
    r31 = float(R[2, 0])
    r32 = float(R[2, 1])
    r33 = float(R[2, 2])
    yaw = np.arctan2(r21, r11)
    pitch = -np.arcsin(r31)
    roll = np.arctan2(r32, r33)
    return np.array([yaw, pitch, roll]).reshape(3, 1)

def eulerAnglesFromQuaternion(q):
    R = rot3DfromQuaternion(q)
    return eulerAnglesFromRot(R)


def constructSuperStateVector(zList, kinStates, extentVector):
    zPerTimeList = [zList[nt].shape[1] for nt in range(len(zList))]

    numZ = 0
    etaList = []
    for nt in range(len(zList)):
        numZ += zPerTimeList[nt]
        pos = kinStates[0:2, nt].reshape((-1, 1))
        psi = float(kinStates[2, nt])
        zArray = zList[nt]
        zBodyArray = rot2D(-psi) @ (zArray - pos)
        etaArray = np.arctan2(zBodyArray[1][:], zBodyArray[0][:]).reshape((1, -1))
        etaList.append(etaArray)
    etaVector = np.concatenate(etaList, axis=None).reshape((-1, 1))

    kinVector = kinStates.transpose().reshape((-1, 1))

    X = np.vstack((etaVector, kinVector, extentVector))

    return X, zPerTimeList


def ransacPlane(listPoints, threshold=0.05, iterations=1000):
    # listPoints: list of 3x1 arrays
    xyz = np.hstack(listPoints).T # shape n_points x 3
    inliers = []
    n_points = xyz.shape[0]
    # pitch and roll based on water plane from RANSAC    n_points = len(xyz)
    i = 1
    while i < iterations:
        idx_samples = random.sample(range(n_points), 3)
        pt0 = xyz[idx_samples[0], :].reshape((-1, ))
        pt1 = xyz[idx_samples[1], :].reshape((-1, ))
        pt2 = xyz[idx_samples[2], :].reshape((-1, ))
        vecA = pt1 - pt0
        vecB = pt2 - pt0
        normal = np.cross(vecA, vecB)
        a, b, c = normal / np.linalg.norm(normal)
        d = -np.sum(normal * pt1)
        distance = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d
                    ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        idx_candidates = np.where(np.abs(distance) <= threshold)[0]
        if len(idx_candidates) > len(inliers):
            equation = [a, b, c, d]
            inliers = idx_candidates
        i += 1
    return equation, inliers

def deconstructSuperStateVector(X, zPerTimeList):
    # TODO: Make it return zList in addition to etaList

    numT = len(zPerTimeList)
    numEta = 0
    indexList = [numEta]
    for neta in range(numT):
        numEta += zPerTimeList[neta]
        indexList.append(numEta)

    etaVector = X[0:numEta]
    kinVector = X[numEta:numEta+numT*6]
    extentVector = X[numEta+numT*6:]

    kinStates = kinVector.reshape((-1, 6)).transpose()

    etaList = []
    for nt in range(numT):
        ind0 = indexList[nt]
        ind1 = indexList[nt+1]
        etaList.append(etaVector[ind0:ind1].reshape((-1, 1)))

    return etaList, kinStates, extentVector


def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.
    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:
    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )
    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:
    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)
    where we define c_{-n} = complex_conjugate(c_{n})
    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.
    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.
    Returns
    -------
    if return_complex == False, the function returns:
    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.
    if return_complex == True, the function returns:
    c : numpy 1-dimensional complex-valued array of size N+1
    """
    # From Shanon theorem we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag

def GradientDescent(f, f_grad, init, alpha=1, tol=1e-5, max_iter=1000):
    """Gradient descent method for unconstraint optimization problem.
    given a starting point x ∈ Rⁿ,
    repeat
        1. Define direction. p := −∇f(x).
        2. Line search. Choose step length α using Armijo Line Search.
        3. Update. x := x + αp.
    until stopping criterion is satisfied.
    Parameters
    --------------------
    f : callable
        Function to be minimized.
    f_grad : callable
        The first derivative of f.
    init : array
        initial value of x.
    alpha : scalar, optional
        the initial value of steplength.
    tol : float, optional
        tolerance for the norm of f_grad.
    max_iter : integer, optional
        maximum number of steps.
    Returns
    --------------------
    xs : array
        x in the learning path
    ys : array
        f(x) in the learning path
    """
    # initialize x, f(x), and f'(x)
    xk = init
    fk = f(xk)
    gfk = f_grad(xk)
    gfk_norm = np.linalg.norm(gfk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    print('Initial condition: y = {:.4f}, x = {} \n'.format(fk, xk))
    # take steps
    while gfk_norm > tol and num_iter < max_iter:
        # determine direction
        pk = -gfk
        # calculate new x, f(x), and f'(x)
        alpha, fk = ArmijoLineSearch(f, xk, pk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * pk
        gfk = f_grad(xk)
        gfk_norm = np.linalg.norm(gfk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
              format(num_iter, fk, xk, gfk_norm))
    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(fk, xk))

    return curve_x, curve_y

def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.
    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    pk : array
        Search direction.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.
    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    derphi0 = float(gfk.T @ pk)
    phi_a0 = f(xk + alpha0 * pk)

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0 * pk)

    return alpha0, phi_a0

def aproxGradient(f, x):
    eps = 1e-6
    J = np.zeros([len(x), 1], dtype=float)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        f1 = f(x1)
        f2 = f(x2)
        J[i] = (f1 - f2) / (2 * eps)
    return J

def aproxJacobian(F, x):
    eps = 1e-6
    Fx = F(x)
    J = np.zeros([len(Fx), len(x)], dtype=float)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        F1 = F(x1)
        F2 = F(x2)
        J[:, i] = (F1 - F2) / (2 * eps)
    return J

def aproxHessian(f, x, lIndexes):
    eps = 1e-6
    H = np.zeros([len(lIndexes), len(lIndexes)], dtype=float)
    for i in range(len(lIndexes)):
        idx1 = lIndexes[i]
        for j in range(len(lIndexes)):
            idx2 = lIndexes[j]
            if idx1 == idx2:
                xp = x.copy()
                xm = x.copy()
                xp[idx1] += eps
                xm[idx1] -= eps
                H[i, i] = (f(xp) - 2 * f(x) + f(xm)) / (eps ** 2)
            else:
                xpp = x.copy()
                xpm = x.copy()
                xmp = x.copy()
                xmm = x.copy()
                xpp[idx1] += eps
                xpm[idx1] += eps
                xmp[idx1] -= eps
                xmm[idx1] -= eps
                xpp[idx2] += eps
                xpm[idx2] -= eps
                xmp[idx2] += eps
                xmm[idx2] -= eps
                fpp = f(xpp)
                fpm = f(xpm)
                fmp = f(xmp)
                fmm = f(xmm)
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
    return H

def isPositiveDefinite(A):
    return np.all(np.linalg.eigvals(A) >= -1e-6)