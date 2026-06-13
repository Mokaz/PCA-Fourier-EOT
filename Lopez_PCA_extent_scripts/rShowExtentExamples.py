import numpy as np
import os
import matplotlib
os.makedirs('fig', exist_ok=True)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Extent import Extent
plt.rcParams.update({'font.size': 16})

# Parameters
plt.close('all')
addTitle = False
saveFigures = True
showFigures = True
L = 1.0
W = 0.4*L
D = 0.6*L
S = 0.2*L
dAngle = np.deg2rad(1)
scales = np.linspace(0.3, 1.0, num=7)

# Extent parameters
param0 = {"type": "ellipsis", "L": L, "W": W}
param1 = {"type": "parabola", "L": L, "W": W}
param2 = {"type": "ellipsisFlatStern", "L": L, "W": W, "D": D, "S": S}
param3 = {"type": "parabolaFlatStern", "L": L, "W": W, "D": D, "S": S}
param4 = {"type": "boxEllipticBow", "L": L, "W": W, "D": D}
param5 = {"type": "boxParabolicBow", "L": L, "W": W, "D": D}
param6 = {"type": "box", "L": L, "W": W, "D": D}
# Extent instances
extent0 = Extent(param0, dAngle)
extent1 = Extent(param1, dAngle)
extent2 = Extent(param2, dAngle)
extent3 = Extent(param3, dAngle)
extent4 = Extent(param4, dAngle)
extent5 = Extent(param5, dAngle)
extent6 = Extent(param6, dAngle)

# Figure: Polar coordinates
plt.figure(figsize=(8, 5))
radii = scales[0] * extent0.radii
plt.plot(np.rad2deg(extent0.angles), radii, linewidth=1.5)
radii = scales[1] * extent1.radii
plt.plot(np.rad2deg(extent1.angles), radii, linewidth=1.5)
radii = scales[2] * extent2.radii
plt.plot(np.rad2deg(extent2.angles), radii, linewidth=1.5)
radii = scales[3] * extent3.radii
plt.plot(np.rad2deg(extent3.angles), radii, linewidth=1.5)
radii = scales[4] * extent4.radii
plt.plot(np.rad2deg(extent4.angles), radii, linewidth=1.5)
radii = scales[5] * extent5.radii
plt.plot(np.rad2deg(extent5.angles), radii, linewidth=1.5)
radii = scales[6] * extent6.radii
plt.plot(np.rad2deg(extent6.angles), radii, linewidth=1.5)
plt.xticks(np.linspace(0, 360, num=5, endpoint=True))
plt.xlabel(r"$\theta$ [deg]")
plt.ylabel(r"$r(\theta)$ [m]")
if addTitle:
    plt.title("Polar coordinates")
if saveFigures:
    plt.savefig("fig/extent_examples_polar_coordinates.eps")

# Figure: Body coordinates
plt.figure(figsize=(8, 5))
bodyCoordinates = scales[0]*extent0.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = scales[1]*extent1.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = scales[2]*extent2.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = scales[3]*extent3.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = scales[4]*extent4.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = scales[5]*extent5.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = scales[6]*extent6.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis([-0.55, 0.55, -0.25, 0.25])
plt.gca().set_aspect('equal', adjustable='box')
#plt.axis("equal")
if addTitle:
    plt.title("Body coordinates")
if saveFigures:
    plt.savefig("fig/extent_examples_body_coordinates.eps")

# Extent vectors using Fourier descriptors
numFourierCoeff = 16
extentFourier0_vec = scales[0]*extent0.getExtentVector("Fourier", True, numFourierCoeff)
extentFourier1_vec = scales[1]*extent1.getExtentVector("Fourier", True, numFourierCoeff)
extentFourier2_vec = scales[2]*extent2.getExtentVector("Fourier", True, numFourierCoeff)
extentFourier3_vec = scales[3]*extent3.getExtentVector("Fourier", True, numFourierCoeff)
extentFourier4_vec = scales[4]*extent4.getExtentVector("Fourier", True, numFourierCoeff)
extentFourier5_vec = scales[5]*extent5.getExtentVector("Fourier", True, numFourierCoeff)
extentFourier6_vec = scales[6]*extent6.getExtentVector("Fourier", True, numFourierCoeff)

# Figure: Fourier series coefficients
plt.figure(figsize=(8, 5))
plt.plot(extentFourier0_vec, marker='o')
plt.plot(extentFourier1_vec, marker='o')
plt.plot(extentFourier2_vec, marker='o')
plt.plot(extentFourier3_vec, marker='o')
plt.plot(extentFourier4_vec, marker='o')
plt.plot(extentFourier5_vec, marker='o')
plt.plot(extentFourier6_vec, marker='o')
plt.xlabel("n ")
plt.ylabel(r"$A_n$ [m]")
if addTitle:
    plt.title("Fourier series coefficients of radius function")
if saveFigures:
    plt.savefig("fig/extent_examples_fourier_coefficients.eps")
# TODO: Plot as scatters with vertical line

# Extent reconstruction using truncated Fourier coefficients
extentFourier0 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier0_vec}, dAngle)
extentFourier1 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier1_vec}, dAngle)
extentFourier2 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier2_vec}, dAngle)
extentFourier3 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier3_vec}, dAngle)
extentFourier4 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier4_vec}, dAngle)
extentFourier5 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier5_vec}, dAngle)
extentFourier6 = Extent({"type": "Fourier", "symmetry": True, "vector": extentFourier6_vec}, dAngle)

# Figure: Fourier reconstruction
plt.figure(figsize=(8, 5))
plt.plot(np.rad2deg(extent0.angles), extentFourier0.radii, linewidth=1.5)
plt.plot(np.rad2deg(extent1.angles), extentFourier1.radii, linewidth=1.5)
plt.plot(np.rad2deg(extent2.angles), extentFourier2.radii, linewidth=1.5)
plt.plot(np.rad2deg(extent3.angles), extentFourier3.radii, linewidth=1.5)
plt.plot(np.rad2deg(extent4.angles), extentFourier4.radii, linewidth=1.5)
plt.plot(np.rad2deg(extent5.angles), extentFourier5.radii, linewidth=1.5)
plt.xlabel(r"$\theta$ [deg]")
plt.ylabel(r"$r(\theta)$ [m]")
if addTitle:
    plt.title("Polar coordinates of Fourier series approximation")
if saveFigures:
    plt.savefig("fig/extent_examples_fourier_reconstruction_polar_coordinates_"
                + str(numFourierCoeff) + ".eps")

#Figure: Fourier extent reconstruction
plt.figure(figsize=(8, 5))
bodyCoordinates = extentFourier0.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = extentFourier1.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = extentFourier2.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = extentFourier3.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = extentFourier4.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = extentFourier5.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
bodyCoordinates = extentFourier6.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis([-0.55, 0.55, -0.25, 0.25])
plt.gca().set_aspect('equal', adjustable='box')
#plt.axis("equal")
if addTitle:
    plt.title("Body coordinates of Fourier series approximation")
if saveFigures:
    plt.savefig("fig/extent_examples_fourier_reconstruction_body_coordinates_"
                + str(numFourierCoeff) + ".eps")

if showFigures:
    plt.show()
