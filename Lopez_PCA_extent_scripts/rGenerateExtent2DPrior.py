import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('fig', exist_ok=True)
os.makedirs('files', exist_ok=True)
from Extent import Extent
plt.rcParams.update({'font.size': 16})

addTitle = False
saveFigures = True
showFigures = True

# Extent prior parameters
priorFileName = "files//extent_prior_fourier_0.npz"
priorFileNamePCA = "files//FourierPCAParameters_0.npz"
dAngle = np.deg2rad(1.0)

lTypes = ["ellipsis", "parabola", "ellipsisFlatStern", "parabolaFlatStern", "boxEllipticBow",
          "boxParabolicBow", "box"]
numSamples = 5
L_array = np.linspace(0.8, 1.2, num=numSamples, endpoint=True)
W_per_L_array = np.linspace(0.2, 0.8, num=numSamples, endpoint=True)
D_per_L_array = np.linspace(0.2, 0.8, num=numSamples, endpoint=True)
S_per_W_array = np.linspace(0.2, 0.8, num=numSamples, endpoint=True)
numFourierCoeff = 64
selectedNumPCAComponents = 4

# generating prior for Fourier coefficients for L=1
extent_vector_list = []
for type in lTypes:
    for L_prior in L_array:
        for W_per_L in W_per_L_array:
            W_prior = W_per_L * L_prior
            for D_per_L in D_per_L_array:
                D_prior = D_per_L * L_prior
                for S_per_W in S_per_W_array:
                    S_prior = S_per_W * W_prior
                    param = {"type": type, "L": L_prior, "W": W_prior, "D": D_prior, "S": S_prior}
                    extent = Extent(param, dAngle)
                    extentFourier_vec = extent.getExtentVector("Fourier", True, numFourierCoeff)
                    extent_vector_list.append([extentFourier_vec])
print(len(extent_vector_list))
extent_vectors = np.squeeze(np.array(extent_vector_list)).transpose()
extent_vector_mean = np.mean(extent_vectors, axis=1).reshape((-1, 1))
extent_vector_covm = np.cov(extent_vectors)

np.savez(priorFileName, mean=extent_vector_mean, covm=extent_vector_covm)

extent_vectors_mean = np.mean(extent_vectors, axis=1).reshape((-1, 1))
extent_vectors_zero_mean = extent_vectors - extent_vectors_mean
extent_vectors_cov = extent_vectors_zero_mean @ extent_vectors_zero_mean.transpose()

print(np.max(np.abs(extent_vector_covm - extent_vectors_cov).reshape((-1, 1))))

eigenvalues, eigenvectors = np.linalg.eig(extent_vectors_cov)
np.savez(priorFileNamePCA, mean=extent_vectors_mean, eigenvectors=eigenvectors)
extent_vectors_corrcoef = np.corrcoef(extent_vectors_zero_mean)

fig = plt.figure(figsize=(8, 5))
im3 = plt.imshow(extent_vectors_corrcoef) #, interpolation='nearest')
fig.colorbar(im3)
if addTitle:
    plt.title("Correlation Coefficients for Fourier Coefficients")
if saveFigures:
    plt.savefig("fig/fourier_components_correlation_coefficients_"
                + str(numFourierCoeff) + ".eps")
    plt.savefig("fig/fourier_components_correlation_coefficients_"
                + str(numFourierCoeff) + ".png")

fig = plt.figure(figsize=(8, 5))
im4 = plt.imshow(np.abs(extent_vectors_corrcoef)) #, interpolation='nearest')
fig.colorbar(im4)
if addTitle:
    plt.title("Absolute Value of Covariance Coefficients for Fourier Coefficients")
if saveFigures:
    plt.savefig("fig/fourier_components_correlation_coefficients_abs_value_"
                + str(numFourierCoeff) + ".eps")
    plt.savefig("fig/fourier_components_correlation_coefficients_abs_value_"
                + str(numFourierCoeff) + ".png")

plt.figure(figsize=(8, 5))
plt.semilogy(eigenvalues)
plt.ylabel("Eigenvalues")
plt.grid()
if addTitle:
    plt.title("Eigenvalues of Covariance Matrix")
if saveFigures:
    plt.savefig("fig/covariance_matrix_eigenvalues_"
                + str(numFourierCoeff) + ".eps")
    plt.savefig("fig/covariance_matrix_eigenvalues_"
                + str(numFourierCoeff) + ".png")

plt.figure(figsize=(8, 5))
extent_vector = extent_vectors_mean.reshape((-1, 1))
extent_eigenvector = Extent({"type": "Fourier", "symmetry": True, "vector": extent_vector}, dAngle)
bodyCoordinates = extent_eigenvector.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5, label="Mean")
for idx in range(2):
    extent_vector = eigenvectors[:, idx].reshape((-1, 1))
    extent_eigenvector = Extent({"type": "Fourier", "symmetry": True, "vector": extent_vector}, dAngle)
    bodyCoordinates = extent_eigenvector.cartesian
    plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5, label="Principal component "+str(idx+1))
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
#plt.legend()
if addTitle:
    plt.title("Extents Associated to Eigenvectors")
if saveFigures:
    plt.savefig("fig/covariance_matrix_eigenvalues_associated_extents_"
                + str(numFourierCoeff) + ".eps")
    plt.savefig("fig/covariance_matrix_eigenvalues_associated_extents_"
                + str(numFourierCoeff) + ".png")

plt.figure(figsize=(8, 5))
for idx in range(10):
    plt.plot(np.abs(eigenvectors[:, idx]), linewidth=1.5, label="Principal component " + str(idx + 1))
plt.ylabel(r"$A_n$")
plt.xlabel("n")
plt.legend()
if addTitle:
    plt.title("")
if saveFigures:
    plt.savefig("fig/covariance_matrix_eigenvalues_abs_value_"
                + str(numFourierCoeff) + ".eps")
    plt.savefig("fig/covariance_matrix_eigenvalues_abs_value_"
                + str(numFourierCoeff) + ".png")
plt.title("Absolute Value of Cosine Fourier Coefficients")

mPCA = extent_vectors_mean
MPCA = eigenvectors
lNumPCAComponents = range(1, numFourierCoeff+1)

# Comparison NPCA vs NFourier
Nparam = 4
paramTrue = {"type": "parabolaFlatStern", "L": 1.0, "W": 0.4, "D": 0.6, "S": 0.2}
extentTrue = Extent(paramTrue, dAngle)
extentTrueVec = extentTrue.getExtentVector("Fourier", True, numFourierCoeff)
extentTrueRadii = extentTrue.radii
Sigma = MPCA[:, 0:Nparam]
PCAparameters = Sigma.transpose() @ (extentTrueVec - mPCA)
extentEstPCAVec = mPCA + Sigma @ PCAparameters
paramEstPCA = {"type": "Fourier", "symmetry": True, "vector": extentEstPCAVec}
extentEstPCA = Extent(paramEstPCA, dAngle)

extentEstFourierVec = extentTrueVec[0:Nparam]
paramEstFourier = {"type": "Fourier", "symmetry": True, "vector": extentEstFourierVec}
extentEstFourier = Extent(paramEstFourier, dAngle)

plt.figure(figsize=(8, 6))
bodyCoordinates = extentTrue.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=2.5, color='black', label="True Extent")
bodyCoordinates = extentEstPCA.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=2.5, linestyle='--', color='tab:orange', label="PCA Approximation")
bodyCoordinates = extentEstFourier.cartesian
plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=2.5, linestyle='-.', color='tab:blue', label="Truncated Fourier")
plt.xlabel("Local y [m]", fontsize=18)
plt.ylabel("Local x [m]", fontsize=18)
plt.legend(fontsize=14, loc='best')
plt.grid(True, linestyle=':', alpha=0.7)
plt.axis("equal")
plt.tight_layout()
if addTitle:
    plt.title("Truncated Fourier vs. PCA Fourier")
if saveFigures:
    plt.savefig("fig/truncated_vs_PCA_presentation.eps", bbox_inches='tight')
    plt.savefig("fig/truncated_vs_PCA_presentation.png", bbox_inches='tight', dpi=300)

# for type in lTypes:
#     worstEstParamTrue = {}
#     worstEstExtentVecEst = np.zeros((numFourierCoeff, 1))
#     aErrorPCAMean = np.zeros((numFourierCoeff, 1))
#     aErrorPCAMax = np.zeros((numFourierCoeff, 1))
#     aErrorFourierMean = np.zeros((numFourierCoeff, 1))
#     aErrorFourierMax = np.zeros((numFourierCoeff, 1))
#     for L_prior in L_array:
#         for W_per_L in W_per_L_array:
#             W_prior = W_per_L * L_prior
#             for D_per_L in D_per_L_array:
#                 D_prior = D_per_L * L_prior
#                 for S_per_W in S_per_W_array:
#                     S_prior = S_per_W * W_prior
#                     paramTrue = {"type": type, "L": L_prior, "W": W_prior, "D": D_prior, "S": S_prior}
#                     print(paramTrue)
#                     extentTrue = Extent(paramTrue, dAngle)
#                     extentTrueVec = extentTrue.getExtentVector("Fourier", True, numFourierCoeff)
#                     extentTrueRadii = extentTrue.radii
#
#                     for nEig in lNumPCAComponents:
#                         Sigma = MPCA[:, 0:nEig]
#                         PCAparameters = Sigma.transpose() @ (extentTrueVec - mPCA)
#                         extentEstPCAVec = mPCA + Sigma @ PCAparameters
#                         paramEstPCA = {"type": "Fourier", "symmetry": True, "vector": extentEstPCAVec}
#                         extentEstPCA = Extent(paramEstPCA, dAngle)
#                         extentEstPCARadii = extentEstPCA.radii
#                         extentEstPCARadiiDiff = np.abs(extentEstPCARadii - extentTrueRadii)
#                         errorPCAMean = np.mean(extentEstPCARadiiDiff)
#                         errorPCAMax = np.max(extentEstPCARadiiDiff)
#
#                         extentEstFourierVec = extentTrueVec[0:nEig]
#                         paramEstFourier = {"type": "Fourier", "symmetry": True, "vector": extentEstFourierVec}
#                         extentEstFourier = Extent(paramEstFourier, dAngle)
#                         extentEstFourierRadii = extentEstFourier.radii
#                         extentEstFourierRadiiDiff = np.abs(extentEstFourierRadii - extentTrueRadii)
#                         errorFourierMean = np.mean(extentEstFourierRadiiDiff)
#                         errorFourierMax = np.max(extentEstFourierRadiiDiff)
#
#                         aErrorFourierMean[nEig - 1] += errorFourierMean
#                         if errorFourierMax > aErrorFourierMax[nEig-1]:
#                             aErrorFourierMax[nEig-1] = errorFourierMax
#
#                         aErrorPCAMean[nEig - 1] += errorPCAMean
#                         if errorPCAMax > aErrorPCAMax[nEig-1]:
#                             aErrorPCAMax[nEig-1] = errorPCAMax
#                             if nEig-1 == selectedNumPCAComponents:
#                                 worstEstParamTrue = paramTrue
#                                 worstEstExtentVecPCA = extentEstPCAVec
#                                 worstEstExtentVecFourier = extentEstFourierVec
#     aErrorPCAMean = aErrorPCAMean/(len(L_array)*len(W_per_L_array)*len(D_per_L_array)*len(S_per_W_array))
#     aErrorFourierMean = aErrorFourierMean / (len(L_array) * len(W_per_L_array) * len(D_per_L_array) * len(S_per_W_array))
#
#     plt.figure()
#     plt.plot(lNumPCAComponents, aErrorPCAMean, label="PCA Mean Error")
#     plt.plot(lNumPCAComponents, aErrorPCAMax, label="PCA Maximum Error")
#     plt.plot(lNumPCAComponents, aErrorFourierMean, label="Fourier Mean Error")
#     plt.plot(lNumPCAComponents, aErrorFourierMax, label="Fourier Maximum Error")
#     plt.xlabel("Number of components")
#     plt.ylabel("Error")
#     plt.ylim((-0.02,0.2))
#     plt.legend()
#     if addTitle:
#         plt.title(type + ": Error")
#     if saveFigures:
#         plt.savefig("fig/error_PCA_" + type + ".eps")
#         plt.savefig("fig/error_PCA_" + type + ".png")
#
#     plt.figure()
#     extentWorstEstTrue = Extent(worstEstParamTrue, dAngle)
#     extentWorstEstPCA = Extent({"type": "Fourier", "symmetry": True, "vector": worstEstExtentVecPCA}, dAngle)
#     extentWorstEstFourier = Extent({"type": "Fourier", "symmetry": True, "vector": worstEstExtentVecFourier}, dAngle)
#     bodyCoordinates = extentWorstEstTrue.cartesian
#     plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5, label="True")
#     bodyCoordinates = extentWorstEstPCA.cartesian
#     plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5, label="PCA Estimate")
#     bodyCoordinates = extentWorstEstFourier.cartesian
#     plt.plot(bodyCoordinates[0, :], bodyCoordinates[1, :], linewidth=1.5, label="Fourier Estimate")
#     plt.xlabel("x [m]")
#     plt.ylabel("y [m]")
#     plt.legend()
#     plt.axis("equal")
#     if addTitle:
#         plt.title(type + ": PCA estimate with largest error")
#     if saveFigures:
#         plt.savefig("fig/error_PCA_max_case_" + type + ".eps")
#         plt.savefig("fig/error_PCA_max_case_" + type + ".png")

if showFigures:
    plt.show()

# TODO: Plot mean with variance interval
# TODO: Plot covariance matrix (normalized)
# TODO: Plot polar and body coordinates of mean with variance interval