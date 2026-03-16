import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def fit(results: dict, plot: bool, return_metadata: bool):
    """
    Fit the interferometric model from systematic scan results using least squares.

    Model
    -----
    O = |Cout @ M @ Cin @ E|^2

    Assumptions
    -----------
    - M is the ideal 4x4 combiner matrix.
    - Cin and Cout are 4x4 complex matrices.
    - E is input-dependent and has ON/OFF complex states per input.
    - Data format is the output of `systematic_scan.run()`:
      results[active_inputs][shifter_idx] -> array (n_outputs, n_phases)

    Parameters
    ----------
    results : dict
        Systematic scan data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        Cout, Cin, Eon, Eoff, metadata
    """
    # Ideal 4x4 MMI matrix
    M = (1 / np.sqrt(4)) * np.array(
        [[1, 1, 1, 1],
         [1, -1j, 1j, 1],
         [1, 1j, -1j, 1],
         [1, -1, -1, 1]],
        dtype=complex,
    )

    # Infer dimensions from data
    first_key = next(iter(results))
    n_shifters = len(results[first_key])
    n_outputs, n_phases = np.asarray(results[first_key][0]).shape
    n_inputs = max((max(k) if len(k) else -1) for k in results.keys()) + 1
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases)

    # Pack/unpack helpers
    nC = 2 * n_outputs * n_inputs
    nE = 2 * n_inputs

    def _unpack(x: np.ndarray):
        c1 = x[:nC].reshape(n_outputs, n_inputs)
        c2 = x[nC:2 * nC].reshape(n_outputs, n_inputs)
        e_on = x[2 * nC:2 * nC + nE]
        e_off = x[2 * nC + nE:2 * nC + 2 * nE]
        Cout = c1 + 1j * c2
        Cin = x[2 * nC + 2 * nE:2 * nC + 2 * nE + n_inputs * n_inputs].reshape(n_inputs, n_inputs)
        Cin = Cin + 1j * x[2 * nC + 2 * nE + n_inputs * n_inputs:].reshape(n_inputs, n_inputs)
        Eon = e_on[:n_inputs] + 1j * e_on[n_inputs:]
        Eoff = e_off[:n_inputs] + 1j * e_off[n_inputs:]
        return Cout, Cin, Eon, Eoff

    def _build_E(active_inputs, shifter_idx, phi):
        E = Eoff.copy()
        for i in active_inputs:
            E[i] = Eon[i]
        E[shifter_idx] *= np.exp(1j * phi)
        return E

    def _predict(Cout, Cin, active_inputs, shifter_idx):
        y = np.zeros((n_outputs, n_phases), dtype=float)
        for k, phi in enumerate(phases):
            E = _build_E(active_inputs, shifter_idx, phi)
            y[:, k] = np.abs(Cout @ M @ Cin @ E) ** 2
        return y

    # Residuals
    keys = list(results.keys())

    def _residuals(x):
        Cout, Cin, Eon, Eoff = _unpack(x)
        r = []
        for active_inputs in keys:
            scans = results[active_inputs]
            for s in range(n_shifters):
                y_meas = np.asarray(scans[s], dtype=float)
                y_pred = _predict(Cout, Cin, active_inputs, s)
                r.append((y_pred - y_meas).ravel())
        # Soft constraints: ||Cin||_2 <= 1, ||Cout||_2 <= 1
        p = []
        for A in (Cin, Cout):
            over = max(0.0, np.linalg.norm(A, 2) - 1.0)
            p.append(np.array([10.0 * over]))
        return np.concatenate(r + p)

    # Initialization: near-identity couplings, simple ON/OFF fields
    rng = np.random.default_rng(0)
    Cout0 = np.eye(n_outputs, n_inputs, dtype=complex) + 1e-2 * (rng.standard_normal((n_outputs, n_inputs)) + 1j * rng.standard_normal((n_outputs, n_inputs)))
    Cin0 = np.eye(n_inputs, dtype=complex) + 1e-2 * (rng.standard_normal((n_inputs, n_inputs)) + 1j * rng.standard_normal((n_inputs, n_inputs)))
    Eon0 = np.ones(n_inputs, dtype=complex)
    Eoff0 = np.zeros(n_inputs, dtype=complex)

    x0 = np.concatenate(
        [
            Cout0.real.ravel(), Cout0.imag.ravel(),
            Eon0.real, Eon0.imag,
            Eoff0.real, Eoff0.imag,
            Cin0.real.ravel(), Cin0.imag.ravel(),
        ]
    )

    sol = least_squares(_residuals, x0, method="trf", max_nfev=300)
    Cout, Cin, Eon, Eoff = _unpack(sol.x)

    metadata = {
        "success": sol.success,
        "cost": sol.cost,
        "message": sol.message,
        "nfev": sol.nfev,
    }
    if plot:
        figs = plot_results(arch, results)
        metadata['figures'] = figs

    if return_metadata:
        return Cout, Cin, Eon, Eoff, metadata
    return Cout, Cin, Eon, Eoff

def plot_results(arch, scan_results, Cout, Cin, Eon, Eoff):
    """
    Plot measured scan data and fitted model predictions.

    Parameters
    ----------
    arch : object
        Architecture object with `n_inputs`, `n_outputs`, and `shifters`.
    scan_results : dict
        Output of `systematic_scan.run()`:
        scan_results[active_inputs][shifter_idx] -> array (n_outputs, n_phases).
    Cout : np.ndarray
        Output coupling matrix.
    Cin : np.ndarray
        Input coupling matrix.
    Eon : np.ndarray
        Complex ON-state input field vector.
    Eoff : np.ndarray
        Complex OFF-state input field vector.

    Returns
    -------
    dict
        Mapping `{f"{k}_inputs": matplotlib.figure.Figure}`.
    """
    M = (1 / np.sqrt(4)) * np.array(
        [[1, 1, 1, 1],
         [1, -1j, 1j, 1],
         [1, 1j, -1j, 1],
         [1, -1, -1, 1]],
        dtype=complex,
    )

    def _predict(active_inputs, shifter_idx, phases):
        y = np.zeros((arch.n_outputs, len(phases)), dtype=float)
        for k, phi in enumerate(phases):
            E = Eoff.copy()
            for i in active_inputs:
                E[i] = Eon[i]
            E[shifter_idx] *= np.exp(1j * phi)
            y[:, k] = np.abs(Cout @ M @ Cin @ E) ** 2
        return y

    figs = {}
    input_combinations = list(scan_results.keys())

    for k in range(arch.n_inputs + 1):
        comb_k_inputs = [c for c in input_combinations if len(c) == k]
        n_rows = len(arch.shifters)
        n_cols = len(comb_k_inputs)

        if n_cols == 0:
            continue

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(max(1, n_cols) * 5, n_rows * 4),
            tight_layout=True,
            squeeze=False
        )

        for c, active_inputs in enumerate(comb_k_inputs):
            for s in range(n_rows):
                y_meas = np.asarray(scan_results[active_inputs][s], dtype=float)
                phases = np.linspace(0.0, 2.0 * np.pi, y_meas.shape[1])
                y_pred = _predict(active_inputs, s, phases)

                ax = axs[s, c]
                for o in range(arch.n_outputs):
                    ax.plot(
                        phases, y_meas[o],
                        linestyle="None", marker="o", markersize=3, alpha=0.7,
                        label=f"Output {o} data" if (s == 0 and c == 0) else None
                    )
                    ax.plot(
                        phases, y_pred[o],
                        linewidth=1.8,
                        label=f"Output {o} model" if (s == 0 and c == 0) else None
                    )

                ax.set_title(f"Inputs: {active_inputs}, Shifter: {s}")
                ax.set_xlabel("Phase [rad]")
                ax.set_ylabel("Flux [ADU]")
                ax.grid(alpha=0.3)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        figs[f"{k}_inputs"] = fig

    return figs