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
         [1, -1j, 1j, -1],
         [1, 1j, -1j, -1],
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
    nC = n_outputs * n_inputs
    nE = 2 * n_inputs

    def _unpack(x: np.ndarray):
        i = 0

        cout_re = x[i:i + nC].reshape(n_outputs, n_inputs)
        i += nC
        cout_im = x[i:i + nC].reshape(n_outputs, n_inputs)
        i += nC

        e_on = x[i:i + nE]
        i += nE
        e_off = x[i:i + nE]
        i += nE

        nCin = n_inputs * n_inputs
        cin_re = x[i:i + nCin].reshape(n_inputs, n_inputs)
        i += nCin
        cin_im = x[i:i + nCin].reshape(n_inputs, n_inputs)

        Cout = cout_re + 1j * cout_im
        Cin = cin_re + 1j * cin_im
        Eon = e_on[:n_inputs] + 1j * e_on[n_inputs:]
        Eoff = e_off[:n_inputs] + 1j * e_off[n_inputs:]
        return Cout, Cin, Eon, Eoff

    def _build_E(Eon, Eoff, active_inputs, shifter_idx, phi):
        E = Eoff.copy()
        for i in active_inputs:
            E[i] = Eon[i]
        E[shifter_idx] *= np.exp(1j * phi)
        return E

    def _predict(Cout, Cin, Eon, Eoff, active_inputs, shifter_idx):
        y = np.zeros((n_outputs, n_phases), dtype=float)
        for k, phi in enumerate(phases):
            E = _build_E(Eon, Eoff, active_inputs, shifter_idx, phi)
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
                y_pred = _predict(Cout, Cin, Eon, Eoff, active_inputs, s)
                r.append((y_pred - y_meas).ravel())
        # Constraints are now handled by explicit boundaries
        return np.concatenate(r)

    # Initialization: near-identity couplings for Cin and Cout
    rng = np.random.default_rng(0)
    Cout0 = np.eye(n_outputs, n_inputs, dtype=complex) + 1e-2 * (rng.standard_normal((n_outputs, n_inputs)) + 1j * rng.standard_normal((n_outputs, n_inputs)))
    Cin0 = np.eye(n_inputs, dtype=complex) + 1e-2 * (rng.standard_normal((n_inputs, n_inputs)) + 1j * rng.standard_normal((n_inputs, n_inputs)))

    # Estimate field scale from global maximum flux to give a better starting point for Eon
    global_max = max(np.max(np.asarray(results[k])) for k in keys)
    # The factor 2 roughly compensates for the 1/sqrt(4) in the M matrix
    e_scale = 2.0 * np.sqrt(global_max)
    
    Eon0 = np.ones(n_inputs, dtype=complex) * e_scale
    Eoff0 = np.zeros(n_inputs, dtype=complex)

    Cout0_flat = np.concatenate([Cout0.real.ravel(), Cout0.imag.ravel()])
    Cin0_flat = np.concatenate([Cin0.real.ravel(), Cin0.imag.ravel()])
    E0_flat = np.concatenate([Eon0.real, Eon0.imag, Eoff0.real, Eoff0.imag])

    # --- Step 1: Fit ONLY Eon and Eoff ---
    # We keep Cin and Cout fixed to our initial identity-like guess
    def _residuals_step1(x_E):
        x_full = np.concatenate([Cout0_flat, x_E, Cin0_flat])
        return _residuals(x_full)

    sol_step1 = least_squares(_residuals_step1, E0_flat, method="trf", max_nfev=200)

    # --- Step 2: Fit all variables ---
    # We use the E arrays obtained from step 1
    x0_full = np.concatenate([Cout0_flat, sol_step1.x, Cin0_flat])

    # Explicit boundaries (Box constraints instead of soft constraints)
    # Constraints on Cin and Cout : elements real/imag parts bounded to stay close to 1
    C_bounds_low = -1.5 * np.ones(len(Cout0_flat))
    C_bounds_high = 1.5 * np.ones(len(Cout0_flat))
    Cin_bounds_low = -1.5 * np.ones(len(Cin0_flat))
    Cin_bounds_high = 1.5 * np.ones(len(Cin0_flat))
    
    # Constraints on Eon and Eoff : unconstrained (well, bounded to infinity to match API)
    E_bounds_low = -np.inf * np.ones(len(E0_flat))
    E_bounds_high = np.inf * np.ones(len(E0_flat))

    bounds_low = np.concatenate([C_bounds_low, E_bounds_low, Cin_bounds_low])
    bounds_high = np.concatenate([C_bounds_high, E_bounds_high, Cin_bounds_high])

    sol = least_squares(_residuals, x0_full, method="trf", bounds=(bounds_low, bounds_high), max_nfev=400)
    Cout, Cin, Eon, Eoff = _unpack(sol.x)

    # Calcul du chi-carré et du chi-carré réduit
    # sol.cost = 0.5 * sum(residuals**2)
    chi_square = 2 * sol.cost
    
    # Calcul du nombre de points de données
    n_data_points = 0
    for active_inputs in keys:
        for s in range(n_shifters):
            n_data_points += np.asarray(results[active_inputs][s]).size
            
    dof = n_data_points - len(sol.x)
    reduced_chi_square = chi_square / dof if dof > 0 else np.inf

    metadata = {
        "success": sol.success,
        "cost": sol.cost,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "dof": dof,
        "message": sol.message,
        "nfev": sol.nfev,
    }
    if plot:
        figs = plot_results(results, Cout, Cin, Eon, Eoff)
        metadata['figures'] = figs

    if return_metadata:
        return Cout, Cin, Eon, Eoff, metadata
    return Cout, Cin, Eon, Eoff

def plot_results(scan_results, Cout, Cin, Eon, Eoff):
    """
    Plot measured scan data and fitted model predictions.

    Parameters
    ----------
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
         [1, -1j, 1j, -1],
         [1, 1j, -1j, -1],
         [1, -1, -1, 1]],
        dtype=complex,
    )

    first_key = next(iter(scan_results))
    n_shifters = len(scan_results[first_key])
    n_outputs = np.asarray(scan_results[first_key][0]).shape[0]
    n_inputs = max((max(k) if len(k) else -1) for k in scan_results.keys()) + 1

    def _predict(active_inputs, shifter_idx, phases):
        y = np.zeros((n_outputs, len(phases)), dtype=float)
        for k, phi in enumerate(phases):
            E = Eoff.copy()
            for i in active_inputs:
                E[i] = Eon[i]
            E[shifter_idx] *= np.exp(1j * phi)
            y[:, k] = np.abs(Cout @ M @ Cin @ E) ** 2
        return y

    figs = {}
    input_combinations = list(scan_results.keys())

    for k in range(n_inputs + 1):
        comb_k_inputs = [c for c in input_combinations if len(c) == k]
        n_rows = n_shifters
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
                for o in range(n_outputs):
                    line_meas, = ax.plot(
                        phases, y_meas[o],
                        linestyle="None", marker="o", markersize=3, alpha=0.7,
                    )
                    color = line_meas.get_color()
                    ax.plot(
                        phases, y_pred[o],
                        color=color,
                        linewidth=1.8,
                    )
                    ax.plot(
                        [], [], color=color, marker="o", markersize=3,
                        linestyle="-", linewidth=1.8, label=f"Output {o}"
                    )

                ax.set_title(f"Inputs: {active_inputs}, Shifter: {s}")
                ax.set_xlabel("Phase [rad]")
                ax.set_ylabel("Flux [ADU]")
                ax.grid(alpha=0.3)
                ax.legend(fontsize="small")

        figs[f"{k}_inputs"] = fig

    return figs