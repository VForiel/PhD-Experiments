import numpy as np
from scipy.optimize import least_squares

def fit_MPDE(scan_data: dict, return_metadata: bool, verbose=False):
    """
    Fit the matrix model of our interferometric test bench using a least-square optimization algorithm.

    Model
    -----
    I = |Cout @ M @ P @ Cin @ E|^2
    I: ndarray[float] of shape (4,) representing the output intensities at the 4 outputs of the chip
    E: ndarray[complex] of shape (4,) representing the input electric fields at the 4 inputs of the chip (each element can be either Eon,i of Eoff,i depending on whether the input i is active or not, knowing that a non-active input is not necessarily 0)
    Din: ndarray[complex] of shape (4, 4) representing the upstream cross-talk between the inputs of the chip (ideally close to identity but can have small non-zero off-diagonal terms and diagonal terms different from 1 to account for small loss of energy)
    P: ndarray[complex] of shape (4, 4) representing the applied phase shifts on the 4 inputs of the chip (diagonal matrix with elements exp(1j * φ_i) where φ_i is the phase shift applied on input i)
    M: ndarray[complex] of shape (4, 4) representing the transfer matrix of the MMI (fixed and known from the design of the chip, considered ideal)
    Dout: ndarray[complex] of shape (4, 4) representing the downstream cross-talk, after the MMI. In practice, due to the matrix degeneracy this matrix also contain imperfections of the MMI itself.
    | |^2 represent the element-wise modulus squared to get output intensities from output electric fields.

    Scan data format
    ----------------
    scan_data = {
        active_inputs (tuple): ndarray of shape (considered_shifter (0->3), considered output (0->3), output_value)
    }
    The size of the last dimension indicate the number of samples taken between 0 and 2*pi for the considered phase shifter.
    """

    # Define the ideal model matrices -----------------------------------------

    # Input electric fields (when all inputs are active)
    Eon = np.array([1, 1, 1, 1], dtype=complex)

    # Input electric fields (when all inputs are inactive)
    Eoff = np.zeros(4, dtype=complex)

    # Upstream cross-talk matrix (spectral norm <= 1 -> allow small loss of energy)
    Din = np.eye(4, dtype=np.complex128)

    # Applied phase shifts
    P = np.eye(4, dtype=np.complex128)

    # MMI transfer matrix (assumed ideal)
    M = (1 / np.sqrt(4)) * np.array(
        [[1, 1, 1, 1],
         [1, -1j, 1j, -1],
         [1, 1j, -1j, -1],
         [1, -1, -1, 1]],
        dtype=complex,
    )

    # We aim to fit the complex parameters of Eon, Eoff, Din and Dout, while P is known from the phase shifter settings during the scan.

    # Model flattening --------------------------------------------------------

    # X vector
    # Our variable will be a vector of [e1, e2, e3, e4, p, s] where 'en' is a boolean indicating whether the input n is active or not, 'p' represent the index of the shifter considered and 's' represent the index on the phase ramp that indicate the phase shift applied on the considered shifter.

    ramp = np.linspace(0, 2 * np.pi, list(scan_data.values())[0].shape[-1])

    # FLatten the input space
    def get_x(inputs:tuple, shifter_index:int, ramp_index:int):
        x = np.zeros(6, dtype=int)
        x[list(inputs)] = 1 # Set active inputs to 1
        x[4] = shifter_index # Set the shifter index
        x[5] = ramp_index # Set the ramp index
        return x
    
    # Get the corresponding true output for a given input and ramp index
    def get_y_true(x):
        # Get active inputs tuple from the first 4 elements
        inputs = tuple(np.where(x[:4]==1)[0].tolist())
        shifter_index = x[4]
        ramp_index = x[5]
        return scan_data[inputs][shifter_index, :, ramp_index]

    # Get the full flattened input and output space for the scan data
    def get_xy_space():
        """
        x_space: ndarray of shape (n_samples, 6)
        y_space: ndarray of shape (n_samples, 4)
        """
        x_space = []
        y_space = []
        for inputs, data in scan_data.items():
            for shifter_index in range(data.shape[0]):
                for ramp_index in range(data.shape[2]):
                    x_space.append(get_x(inputs, shifter_index, ramp_index))
                    y_space.append(get_y_true(x_space[-1]))
        return np.array(x_space), np.array(y_space)

    x_space, y_space = get_xy_space()

    def pack_params(Eon, Eoff, Din):

        Eon_flat = Eon.flatten()
        Eon_real = np.real(Eon_flat)
        Eon_imag = np.imag(Eon_flat)[1:]  # We can fix the phase of the first input as a reference to remove degeneracy

        Eoff_flat = Eoff.flatten()
        Eoff_real = np.real(Eoff_flat)
        Eoff_imag = np.imag(Eoff_flat)[1:]  # We can fix the phase of the first input as a reference to remove degeneracy

        # Crosstalk matrices are stored as a real vector so `least_squares`

        # Keep only non diagonal and non first-input phase terms
        # Diagonal phase terms are degenerated with E and first-input phase is used as a reference
        D_image_mask = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ])

        Din_real = np.real(Din).flatten()
        Din_imag = (np.imag(Din) * D_image_mask).flatten()

        return np.concatenate([
            Eon_real, Eon_imag,
            Eoff_real, Eoff_imag,
            Din_real, Din_imag,
        ])

    def unpack_params(params):

        Eon_real = params[0:4]
        Eon_imag = np.concatenate([[0], params[4:7]])
        Eon = Eon_real + 1j * Eon_imag

        Eoff_real = params[7:11]
        Eoff_imag = np.concatenate([[0], params[11:14]])
        Eoff = Eoff_real + 1j * Eoff_imag

        D_image_mask = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ])

        Din_real = params[14:30].reshape(4, 4)
        Din_imag = np.zeros((4, 4)) # Initialize with zeros
        Din_imag[D_image_mask == 1] = params[30:39] # Fill only the non-diagonal and non first-input phase terms
        Din = Din_real + 1j * Din_imag

        return Eon, Eoff, Din

    # Cost function for least squares optimization ----------------------------

    def unique_residual(x, y_true, params):

        Eon, Eoff, Din = unpack_params(params)

        # Build E
        inputs = x[:4].astype(bool)  # First 4 elements indicate active inputs
        E = Eoff.copy()
        E[inputs] = Eon[inputs]  # Set active inputs to their "on" value

        # Build P
        phases = np.zeros(4)  # Initialize phase shifts to zero
        shifter_index = x[4]
        ramp_index = x[5]
        phases[shifter_index] = ramp[ramp_index]  # Set the phase shift for the considered shifter
        P = np.diag(np.exp(1j * phases))  # Update phase shift matrix

        # Compute the predicted output
        y_pred = np.abs(M @ P @ Din @ E)**2

        # Compute the cost as the sum of squared differences between predicted and true outputs
        return y_pred - y_true

    def global_residuals(params):
        res = np.empty(x_space.shape[0] * 4, dtype=float)
        k = 0
        for x, y_true in zip(x_space, y_space):
            r = unique_residual(x, y_true, params)   # shape (4,)
            res[k:k+4] = r
            k += 4
        return res

    # Least squares optimization ----------------------------------------------

    x0 = pack_params(Eon, Eoff, Din)

    result = least_squares(
        fun=global_residuals,
        x0=x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
        verbose=verbose,
    )

    Eon_fit, Eoff_fit, Din_fit = unpack_params(result.x)

    def model(active_inputs_tuple, phases_array):
        E = Eoff.copy()
        E[list(active_inputs_tuple)] = Eon_fit[list(active_inputs_tuple)]
        P = np.diag(np.exp(1j * phases_array))
        return np.abs(M @ P @ Din_fit @ E)**2

    if return_metadata:
        return {
            "model": model,
            "Eon": Eon_fit,
            "Eoff": Eoff_fit,
            "Din": Din_fit,
            "cost": result.cost,
            "optimality": result.optimality,
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
        }

    return model, Eon_fit, Eoff_fit, Din_fit
