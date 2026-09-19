import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PARAMETERS
# ============================================================

IMAGE_PATH = "image.png"       # <-- put your image filename here

N = 8
# Image size will be 2^N x 2^N
# N = 8 -> 256 x 256

CHI_MAX = 8
# Fixed maximum MPS bond dimension

MAX_SWEEPS = 20
# One sweep = left -> right followed by right -> left

TOLERANCE = 1e-7
# Stop if relative improvement becomes smaller than this


# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("L")

size = 2**N

image = image.resize(
    (size, size),
    Image.Resampling.LANCZOS
)

image_array = np.asarray(image, dtype=float)

print("Image shape:", image_array.shape)
print("Number of pixels:", image_array.size)


# ============================================================
# 2. CONVERT IMAGE TO RG / QUADRANT TENSOR
# ============================================================
#
# Each pixel gets N quadrant labels.
#
# quadrant:
#
#   0 = top-left
#   1 = top-right
#   2 = bottom-left
#   3 = bottom-right
#
# Therefore
#
# 2^N x 2^N image
#
# becomes
#
# 4 x 4 x ... x 4
#
# with N indices.
#
# ============================================================


def image_to_rg_tensor(image):

    size = image.shape[0]

    N = int(np.log2(size))

    tensor = np.zeros([4] * N, dtype=float)

    for row in range(size):

        for col in range(size):

            indices = []

            for level in range(N):

                shift = N - 1 - level

                row_bit = (row >> shift) & 1
                col_bit = (col >> shift) & 1

                quadrant = 2 * row_bit + col_bit

                indices.append(quadrant)

            tensor[tuple(indices)] = image[row, col]

    return tensor


rg_tensor = image_to_rg_tensor(image_array)

print("\nRG tensor shape:")
print(rg_tensor.shape)


# ============================================================
# 3. INITIAL MPS USING TRUNCATED SVD
# ============================================================
#
# This gives us an initial guess for the optimized MPS.
#
# The IMPORTANT point:
#
# We are NOT stopping here.
#
# The first method would stop after this TT-SVD.
#
# The second method then optimizes all these MPS tensors.
#
# ============================================================


def tensor_to_mps_svd(tensor, chi_max):

    dims = tensor.shape

    N = len(dims)

    cores = []

    left_rank = 1

    working_tensor = tensor.copy()

    for site in range(N - 1):

        physical_dim = dims[site]

        matrix = working_tensor.reshape(
            left_rank * physical_dim,
            -1
        )

        U, S, Vh = np.linalg.svd(
            matrix,
            full_matrices=False
        )

        chi = min(
            chi_max,
            len(S)
        )

        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]

        core = U.reshape(
            left_rank,
            physical_dim,
            chi
        )

        cores.append(core)

        working_tensor = S[:, None] * Vh

        left_rank = chi

    final_core = working_tensor.reshape(
        left_rank,
        dims[-1],
        1
    )

    cores.append(final_core)

    return cores


mps = tensor_to_mps_svd(
    rg_tensor,
    CHI_MAX
)


# ============================================================
# 4. RECONSTRUCT A FULL TENSOR FROM AN MPS
# ============================================================


def mps_to_tensor(cores):

    tensor = cores[0]

    for core in cores[1:]:

        tensor = np.tensordot(
            tensor,
            core,
            axes=([-1], [0])
        )

    tensor = np.squeeze(
        tensor,
        axis=(0, -1)
    )

    return tensor


# ============================================================
# 5. ERROR FUNCTION
# ============================================================
#
# This is exactly the quantity we want to minimize:
#
# || target - MPS ||^2
#
# ============================================================


def squared_error(target, cores):

    approximation = mps_to_tensor(cores)

    difference = target - approximation

    return np.sum(difference**2)


# ============================================================
# INITIAL ERROR BEFORE OPTIMIZATION
# ============================================================

initial_tensor = mps_to_tensor(mps)

initial_error = squared_error(
    rg_tensor,
    mps
)

print("\nInitial truncated-SVD error:")
print(initial_error)


# ============================================================
# 6. LEFT ENVIRONMENT
# ============================================================
#
# Suppose we are optimizing core k.
#
# Everything to the LEFT of core k is contracted into
# one matrix L.
#
#          L -- Gamma[k] -- R
#
# L contains the effect of all earlier MPS tensors.
#
# ============================================================


def left_environment(cores, k):

    L = np.ones((1, 1))

    for site in range(k):

        core = cores[site]

        # L[a,l] * G[l,i,r]
        #
        # -> tensor indexed by
        #
        # a, i, r

        L = np.einsum(
            "al,lir->air",
            L,
            core
        )

        # Combine all physical indices into one
        L = L.reshape(
            -1,
            core.shape[2]
        )

    return L


# ============================================================
# 7. RIGHT ENVIRONMENT
# ============================================================
#
# Everything to the RIGHT of core k is contracted into R.
#
#          L -- Gamma[k] -- R
#
# ============================================================


def right_environment(cores, k):

    R = np.ones((1, 1))

    for site in range(
        len(cores) - 1,
        k,
        -1
    ):

        core = cores[site]

        # G[l,i,r] * R[r,b]

        R = np.einsum(
            "lir,rb->lib",
            core,
            R
        )

        R = R.reshape(
            core.shape[0],
            -1
        )

    return R


# ============================================================
# 8. OPTIMIZE ONE MPS TENSOR
# ============================================================
#
# THIS IS THE HEART OF THE PAPER'S SECOND METHOD.
#
# Freeze every MPS core except core k.
#
# The approximation becomes
#
#
#          X ~= L G R
#
#
# where:
#
#       X = original image tensor
#       L = everything left of G
#       G = tensor we are optimizing
#       R = everything right of G
#
#
# We minimize
#
#       || X - L G R ||^2
#
#
# For each physical index i this gives the normal equation
#
#
#   (L^T L) G_i (R R^T)
#
#             =
#
#        L^T X_i R^T
#
#
# which is the least-squares equivalent of the
# B Gamma = E equation used in the paper.
#
# ============================================================


def optimize_one_core(
    target,
    cores,
    k,
    rcond=1e-12
):

    # --------------------------------------------------------
    # Build left and right environments
    # --------------------------------------------------------

    L = left_environment(
        cores,
        k
    )

    R = right_environment(
        cores,
        k
    )

    core = cores[k]

    left_rank = core.shape[0]
    physical_dim = core.shape[1]
    right_rank = core.shape[2]

    # --------------------------------------------------------
    # Reshape original full tensor so that
    #
    #      target[left, physical, right]
    #
    # --------------------------------------------------------

    left_size = int(
        np.prod(target.shape[:k])
    ) if k > 0 else 1

    right_size = int(
        np.prod(target.shape[k + 1:])
    ) if k < target.ndim - 1 else 1

    X = target.reshape(
        left_size,
        physical_dim,
        right_size
    )

    # --------------------------------------------------------
    # Construct normal-equation matrices
    #
    # A = L^T L
    # B = R R^T
    #
    # --------------------------------------------------------

    A = L.T @ L

    B = R @ R.T

    # Pseudoinverse is more numerically stable than
    # explicitly calculating an ordinary inverse.

    A_inverse = np.linalg.pinv(
        A,
        rcond=rcond
    )

    B_inverse = np.linalg.pinv(
        B,
        rcond=rcond
    )

    new_core = np.zeros_like(core)

    # --------------------------------------------------------
    # Optimize separately for each physical index
    # --------------------------------------------------------

    for i in range(physical_dim):

        X_i = X[:, i, :]

        E = (
            L.T
            @ X_i
            @ R.T
        )

        # Solve:
        #
        # A G_i B = E
        #
        # therefore
        #
        # G_i = A^-1 E B^-1

        G_i = (
            A_inverse
            @ E
            @ B_inverse
        )

        new_core[:, i, :] = G_i

    cores[k] = new_core


# ============================================================
# 9. SWEEP OPTIMIZATION
# ============================================================
#
# Perform:
#
# core 1 -> core 2 -> ... -> core N
#
# followed by
#
# core N -> ... -> core 2 -> core 1
#
# Then repeat.
#
# ============================================================


def optimize_mps(
    target,
    cores,
    max_sweeps=20,
    tolerance=1e-7
):

    error_history = []

    previous_error = squared_error(
        target,
        cores
    )

    error_history.append(
        previous_error
    )

    print("\nStarting sweep optimization")
    print("---------------------------")

    print(
        f"Initial error = "
        f"{previous_error:.6e}"
    )

    N = len(cores)

    for sweep in range(
        1,
        max_sweeps + 1
    ):

        # ====================================================
        # LEFT -> RIGHT SWEEP
        # ====================================================

        for k in range(N):

            optimize_one_core(
                target,
                cores,
                k
            )

        # ====================================================
        # RIGHT -> LEFT SWEEP
        # ====================================================

        for k in range(
            N - 1,
            -1,
            -1
        ):

            optimize_one_core(
                target,
                cores,
                k
            )

        # ====================================================
        # CALCULATE NEW ERROR
        # ====================================================

        current_error = squared_error(
            target,
            cores
        )

        error_history.append(
            current_error
        )

        improvement = (
            previous_error
            - current_error
        )

        relative_improvement = (
            improvement
            / max(previous_error, 1e-30)
        )

        print(
            f"Sweep {sweep:2d}: "
            f"error = {current_error:.6e}, "
            f"relative improvement = "
            f"{relative_improvement:.3e}"
        )

        # ====================================================
        # CONVERGENCE TEST
        # ====================================================

        if (
            relative_improvement >= 0
            and
            relative_improvement < tolerance
        ):

            print(
                "\nConverged."
            )

            break

        previous_error = current_error

    return cores, error_history


# ============================================================
# 10. RUN OPTIMIZATION
# ============================================================


optimized_mps, error_history = optimize_mps(
    rg_tensor,
    mps,
    max_sweeps=MAX_SWEEPS,
    tolerance=TOLERANCE
)


# ============================================================
# 11. RECONSTRUCT OPTIMIZED RG TENSOR
# ============================================================


optimized_tensor = mps_to_tensor(
    optimized_mps
)


# ============================================================
# 12. CONVERT RG TENSOR BACK TO IMAGE
# ============================================================


def rg_tensor_to_image(tensor):

    N = tensor.ndim

    size = 2**N

    image = np.zeros(
        (size, size),
        dtype=float
    )

    for row in range(size):

        for col in range(size):

            indices = []

            for level in range(N):

                shift = N - 1 - level

                row_bit = (
                    row >> shift
                ) & 1

                col_bit = (
                    col >> shift
                ) & 1

                quadrant = (
                    2 * row_bit
                    + col_bit
                )

                indices.append(
                    quadrant
                )

            image[row, col] = (
                tensor[
                    tuple(indices)
                ]
            )

    return image


# ============================================================
# 13. GET BOTH RECONSTRUCTIONS
# ============================================================
#
# We compare:
#
# 1. Ordinary truncated SVD
# 2. Optimized MPS
#
# ============================================================


svd_image = rg_tensor_to_image(
    initial_tensor
)

optimized_image = rg_tensor_to_image(
    optimized_tensor
)

svd_image = np.clip(
    svd_image,
    0,
    255
)

optimized_image = np.clip(
    optimized_image,
    0,
    255
)


# ============================================================
# 14. QUALITY METRICS
# ============================================================


def calculate_metrics(
    original,
    approximation
):

    mse = np.mean(
        (
            original
            - approximation
        )**2
    )

    rmse = np.sqrt(mse)

    if mse == 0:

        psnr = np.inf

    else:

        psnr = (
            10
            * np.log10(
                255**2 / mse
            )
        )

    return mse, rmse, psnr


svd_mse, svd_rmse, svd_psnr = (
    calculate_metrics(
        image_array,
        svd_image
    )
)

opt_mse, opt_rmse, opt_psnr = (
    calculate_metrics(
        image_array,
        optimized_image
    )
)


print("\n===================================")
print("RESULTS")
print("===================================")

print("\nTruncated SVD:")
print("MSE  =", svd_mse)
print("RMSE =", svd_rmse)
print("PSNR =", svd_psnr, "dB")

print("\nOptimized MPS:")
print("MSE  =", opt_mse)
print("RMSE =", opt_rmse)
print("PSNR =", opt_psnr, "dB")

print(
    "\nPSNR improvement =",
    opt_psnr - svd_psnr,
    "dB"
)


# ============================================================
# 15. STORAGE
# ============================================================


original_parameters = (
    image_array.size
)

mps_parameters = sum(
    core.size
    for core in optimized_mps
)

compression_ratio = (
    original_parameters
    / mps_parameters
)

print("\nStorage")
print("-------")

print(
    "Original parameters:",
    original_parameters
)

print(
    "MPS parameters:",
    mps_parameters
)

print(
    "Compression ratio:",
    compression_ratio
)


# ============================================================
# 16. DISPLAY THREE IMAGES
# ============================================================


plt.figure(
    figsize=(15, 5)
)


plt.subplot(1, 3, 1)

plt.imshow(
    image_array,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Original image"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    svd_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Truncated SVD\n"
    f"χ = {CHI_MAX}, "
    f"PSNR = {svd_psnr:.2f} dB"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    optimized_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Sweep-optimized MPS\n"
    f"χ = {CHI_MAX}, "
    f"PSNR = {opt_psnr:.2f} dB"
)

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 17. PLOT OPTIMIZATION CONVERGENCE
# ============================================================


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(len(error_history)),
    error_history,
    marker="o"
)

plt.yscale("log")

plt.xlabel(
    "Sweep number"
)

plt.ylabel(
    r"$||\psi-\tilde{\psi}||^2$"
)

plt.title(
    "MPS optimization convergence"
)

plt.grid()

plt.tight_layout()

plt.show()