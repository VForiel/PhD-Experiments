import numpy as np
from scipy.optimize import least_squares, curve_fit
import matplotlib.pyplot as plt

from .model_fitting.cmpce import fit_CMPCE
from .model_fitting.mpce import fit_MPCE
from .model_fitting.mpe import fit_MPE
from .model_fitting.mpde import fit_MPDE
from .model_fitting.dmpde import fit_DMPDE
from .model_fitting.rmpre import fit_RMPRE
from .model_fitting.mpre import fit_MPRE
from .model_fitting.dmpde_staged import fit_DMPDE_staged

# def plot_fit(scan_data, model):
#     """
#     Plot the scan data and the fitted model for visual comparison.
#     Each plot correspond to a certain number of active inputs.
#     On each plot, the subplot columns correspond to the active inputs combination.
#     The subplots rows correspond to the considered shifter.
#     On each subplot, the colored shaped points (data) and the colored line (model) correspond to a specific output.
#     """

#     ramp = np.linspace(0, 2 * np.pi, scan_data[()].shape[-1]) 

def load_scan_data(file_path):
    """
    Load the scan data from a .npy file and return it as a dictionary.
    The .npy file should contain a dictionary where keys are tuples of active input indices,
    and values are measured intensity arrays of shape (n_shifters, n_outputs, n_ramp_samples).
    """
    scan_data = {}
    data = np.load(file_path)
    scan_data = {eval(key): data[key] for key in data.files}

    # scan_data_filtered = {}
    # for key, value in scan_data.items():
    #     if len(key) in [1,]:  # Keep only configurations with 1 or 2 active inputs
    #         scan_data_filtered[key] = value
    # return scan_data_filtered

    return scan_data

def modulation(phase, amplitude, offset, phase_shift):
    """
    Simple sinusoidal modulation function.
    """
    return offset + amplitude * np.sin(phase + phase_shift)

def get_modulations(scan_data: dict):
    """
    Extract modulations from scan data.
    """
    modulations_params = {}
    smoothed_data = {}
    for config, data in scan_data.items():
        n_shifters, n_outputs, n_ramp = data.shape
        ramp = np.linspace(0, 2 * np.pi, n_ramp)

        params = np.zeros((n_shifters, n_outputs, 3))  # A, O, P for each shifter-output pair
        data_smoothed = np.zeros_like(data)
        for shifter_idx in range(n_shifters):
            for output_idx in range(n_outputs):
                measured = data[shifter_idx, output_idx, :]
                popt, _ = curve_fit(
                    lambda phase, A, O, P: modulation(phase, A, O, P),
                    ramp,
                    measured,
                    p0=[(measured.max() - measured.min()) / 2, measured.mean(), 0],
                )
                params[shifter_idx, output_idx, :] = popt
                data_smoothed[shifter_idx, output_idx, :] = modulation(ramp, *popt)
        modulations_params[config] = params
        smoothed_data[config] = data_smoothed
    return modulations_params, smoothed_data

def plot_fit(scan_data: dict, model, modulations_params=None):
    """
    Plot measured vs predicted intensities for each scan configuration.

    Parameters
    ----------
    scan_data : dict
        Dictionary where keys are tuples of active input indices,
        and values are measured intensity arrays of shape (n_shifters, n_outputs, n_ramp_samples).
    model : callable
        Function that takes (active_inputs_tuple, phases_array) and returns
        predicted intensities of shape (n_outputs,).
        phases_array is a 1D array of length 4 where each element is the phase
        applied to the corresponding shifter.

    Returns
    -------
    figures : list of tuple
        One ``(fig, axes)`` pair per active-input configuration.
        Each figure contains one subplot per shifter.
    """

    # Extract dimensions from first measurement
    first_data = next(iter(scan_data.values()))
    n_shifters, n_outputs, n_ramp = first_data.shape

    # Phase ramp used in measurements
    ramp = np.linspace(0, 2 * np.pi, n_ramp)

    # Color palette for the 4 outputs
    colors = plt.cm.tab10(np.arange(n_outputs))

    figures = []

    # Iterate over each configuration (active inputs)
    for config_idx, (active_inputs, measured_data) in enumerate(scan_data.items()):
        # Create one figure per active-input configuration so the scan is easier to compare.
        fig, axes = plt.subplots(
            1,
            n_shifters,
            figsize=(4 * n_shifters, 5),
            tight_layout=True,
        )

        if n_shifters == 1:
            axes = np.array([axes])

        fig.suptitle(f"Active inputs: {active_inputs}", fontsize=14)

        # Iterate over each shifter
        for shifter_idx in range(n_shifters):
            ax = axes[shifter_idx]

            # Generate predictions for all phases in the ramp
            predicted_all = []
            for phase_idx, phase_value in enumerate(ramp):
                # Build phase vector: only shifter_idx is non-zero
                phases_vector = np.zeros(n_shifters)
                phases_vector[shifter_idx] = phase_value

                # Get model prediction for this configuration and phase vector
                y_pred = model(active_inputs, phases_vector)  # Expected shape: (n_outputs,)
                predicted_all.append(y_pred)

            predicted_all = np.array(predicted_all)  # Shape: (n_ramp, n_outputs)
            measured = measured_data[shifter_idx, :, :]  # Shape: (n_outputs, n_ramp)

            # Plot each output with measured (solid) vs predicted (dashed)
            for output_idx in range(n_outputs):
                ax.plot(
                    ramp,
                    measured[output_idx, :],
                    "o",
                    color=colors[output_idx],
                    label=f"Out {output_idx} (meas)",
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.6,
                )
                ax.plot(
                    ramp,
                    modulation(ramp, *modulations_params[active_inputs][shifter_idx, output_idx]),
                    "-",
                    color=colors[output_idx],
                    label=f"Out {output_idx} (mod)",
                    linewidth=1.5,
                    alpha=0.9,
                )
                ax.plot(
                    ramp,
                    predicted_all[:, output_idx],
                    "--",
                    color=colors[output_idx],
                    label=f"Out {output_idx} (pred)",
                    linewidth=1.5,
                    alpha=0.9,
                )

            # Formatting
            ax.set_xlabel("Phase (rad)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title(f"Shifter {shifter_idx}")
            ax.grid(True, alpha=0.3)
            # ax.legend(fontsize=8, loc="best")
            ax.set_xlim([0, 2 * np.pi])
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

        figures.append((fig, axes))

    if len(figures) == 1:
        return figures[0]

    return figures


if __name__ == "__main__":

    data_path = "D:/PhD-Experiments/highlighted_results/2026-03-18_b38b520/2_N4x4-T8_characterisation/systematic_scan.npz"
    scan_data = load_scan_data(data_path)
    metadata = fit_MPE(scan_data, plot=True, return_metadata=False)
    print(metadata['M'])