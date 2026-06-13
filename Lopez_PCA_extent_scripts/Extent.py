import numpy as np
from tools import ssa, cart2pol, pol2cart

# TODO: Add functions for boundary point. See constraintsTools.py

class Extent:

    def __init__(self, parameters, dAngle):
        # angles for polar representation
        angles = np.arange(0.0, 2*np.pi, dAngle)
        self.angles = angles

        # radii for polar representation
        nAngles = angles.size

        if parameters["type"] == "ellipsis":
            L = parameters["L"]
            W = parameters["W"]
            x_interpol = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_interpol = (W / 2) * np.sqrt(1 - ((2 / L) * x_interpol)**2)
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            radii = np.interp(np.abs(ssa(angles)), angles_interpol, r_interpol, period=2*np.pi)

        elif parameters["type"] == "parabola":
            L = parameters["L"]
            W = parameters["W"]
            x_interpol = np.linspace(-L / 2, L / 2, num=nAngles, endpoint=True)
            y_interpol = (W / 2) - (2 * W / L**2) * x_interpol**2
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            radii = np.interp(np.abs(ssa(angles)), angles_interpol, r_interpol, period=2*np.pi)

        elif parameters["type"] == "ellipsisFlatStern":
            L = parameters["L"]
            W = parameters["W"]
            D = parameters["D"]
            S = parameters["S"]
            m = -W**2 / 4 / D**2
            n = -W**2 / 2 / D
            p = -(W**2 - S**2) / 4 / (L - D)**2
            q = (W**2 - S**2) / 2 / (L - D)
            x1 = np.linspace(-L / 2, (L / 2 - D), num=nAngles, endpoint=False)
            x2 = np.linspace((L / 2 - D), L / 2, num=nAngles, endpoint=True)
            x_interpol = np.concatenate((x1, x2), axis=0)
            y1 = p * (x1 + L / 2)**2 + q * (x1 + L / 2) + (S**2) / 4
            y2 = m * (x2 - L / 2)**2 + n * (x2 - L / 2)
            y_interpol = np.concatenate((y1, y2), axis=0)
            y_interpol = np.sqrt(y_interpol)
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            theta_S = np.arctan2(S, -L)
            angles_0 = np.abs(ssa(angles))
            radii = np.zeros(angles_0.shape)
            for indx, angle in enumerate(angles_0):
                if angle <= theta_S:
                    radii[indx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                else:
                    radii[indx] = -L / 2 / np.cos(angle)

        elif parameters["type"] == "parabolaFlatStern":
            L = parameters["L"]
            W = parameters["W"]
            D = parameters["D"]
            S = parameters["S"]
            m = -W / 2 / D**2
            n = -W / D
            p = -(W - S) / 2 / (L - D)**2
            q = (W - S) / (L - D)
            x1 = np.linspace(-L / 2, (L / 2 - D), num=nAngles, endpoint=False)
            x2 = np.linspace((L / 2 - D), L / 2, num=nAngles, endpoint=True)
            x_interpol = np.concatenate((x1, x2), axis=0)
            y1 = p * (x1 + L / 2)**2 + q * (x1 + L / 2) + S / 2
            y2 = m * (x2 - L / 2)**2 + n * (x2 - L / 2)
            y_interpol = np.concatenate((y1, y2), axis=0)
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            theta_S = np.arctan2(S, -L)
            angles_0 = np.abs(ssa(angles))
            radii = np.zeros(angles_0.shape)
            for indx, angle in enumerate(angles_0):
                if angle <= theta_S:
                    radii[indx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                else:
                    radii[indx] = -L / 2 / np.cos(angle)

        elif parameters["type"] == "boxEllipticBow":
            L = parameters["L"]
            W = parameters["W"]
            D = parameters["D"]
            x_interpol = np.linspace(L / 2 - D, L / 2, num=nAngles, endpoint=True)
            y_interpol = W / 2 * np.sqrt(1 - ((x_interpol - (L / 2 - D)) / D)**2)
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            theta_1 = np.arctan2(W / 2, L / 2 - D)
            theta_2 = np.arctan2(W, -L)
            angles_0 = np.abs(ssa(angles))
            radii = np.zeros(angles_0.shape)
            for indx, angle in enumerate(angles_0):
                if angle <= theta_1:
                    radii[indx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= theta_2:
                    radii[indx] = W / 2 / np.sin(angle)
                else:
                    radii[indx] = -L / 2 / np.cos(angle)


        elif parameters["type"] == "boxParabolicBow":
            L = parameters["L"]
            W = parameters["W"]
            D = parameters["D"]
            x_interpol = np.linspace(L / 2 - D, L / 2, num=nAngles, endpoint=True)
            y_interpol = -W / 2 / D**2 * (x_interpol - (L / 2 - D))**2 + W / 2
            angles_interpol, r_interpol = cart2pol(x_interpol, y_interpol)
            theta_1 = np.arctan2(W / 2, L / 2 - D)
            theta_2 = np.arctan2(W, -L)
            angles_0 = np.abs(ssa(angles))
            radii = np.zeros(angles_0.shape)
            for indx, angle in enumerate(angles_0):
                if angle <= theta_1:
                    radii[indx] = np.interp(angle, angles_interpol, r_interpol, period=2*np.pi)
                elif angle <= theta_2:
                    radii[indx] = W / 2 / np.sin(angle)
                else:
                    radii[indx] = -L / 2 / np.cos(angle)

        elif parameters["type"] == "box":
            L = parameters["L"]
            W = parameters["W"]
            theta_0 = np.arctan2(W, L)
            angles_0 = np.abs(ssa(angles))
            radii = np.zeros(angles_0.shape)
            for indx, angle in enumerate(angles_0):
                if angle <= theta_0:
                    radii[indx] = L / 2 / np.cos(angle)
                elif angle <= np.pi - theta_0:
                    radii[indx] = W / 2 / np.cos(np.pi/2 - angle)
                else:
                    radii[indx] = L / 2 / np.cos(np.pi - angle)

        elif parameters["type"] == "Fourier":
            extent_vector = parameters["vector"]
            if parameters["symmetry"]:
                N_e = extent_vector.size
                extent_vector = np.concatenate((extent_vector, np.zeros((1, N_e-1))), axis=None)
            else:
                N_e = int((extent_vector.size + 1)/2)
            extent_vector = extent_vector.reshape((-1, 1))
            N_r = angles.size
            B = np.zeros((N_r, 2 * N_e - 1))
            for nr in range(N_r):
                for ne in range(N_e):
                    if ne == 0:
                        B[nr][ne] = 0.5
                    else:
                        angle = angles[nr]
                        B[nr][ne] = np.cos(ne * angle)
                        B[nr][ne + N_e - 1] = np.sin(ne * angle)
            radii = np.matmul(B, extent_vector)

        self.radii = radii.reshape((-1, ))

        cart_x, cart_y = pol2cart(self.angles, self.radii)
        cart_x_closed = np.concatenate((cart_x, cart_x[0]), axis=None).reshape((1, -1))
        cart_y_closed = np.concatenate((cart_y, cart_y[0]), axis=None).reshape((1, -1))
        cart = np.concatenate((cart_x_closed, cart_y_closed), axis=0)
        self.cartesian = cart

    def getExtentVector(self, type, symmetry, Nvector=10): #TODO: Nvector should be equal to number of Fourier coefficients
        if type == "Fourier":
            f_sample = 2*Nvector
            angles_fft = np.linspace(0, 2*np.pi, f_sample, endpoint=False)
            r_fft = np.interp(angles_fft, self.angles, self.radii, period=2*np.pi)
            y = np.fft.rfft(r_fft)/angles_fft.size
            y *= 2
            a0 = y[0].real
            a = y[1:-1].real
            b = -y[-1:1].imag
            if symmetry:
                return np.concatenate((a0, a), axis=None).reshape((-1, 1))
            else:
                return np.concatenate((a0, a, b), axis=None).reshape((-1, 1))
        else:
            if symmetry:
                angles = np.linspace(0, np.pi, num=Nvector, endpoint=False)
            else:
                angles = np.linspace(0, 2*np.pi, num=Nvector, endpoint=False)
            return np.interp(angles, self.angles, self.radii, period=2*np.pi).reshape((-1, 1))
