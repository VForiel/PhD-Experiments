from itertools import combinations
import numpy as np
import phobos
import matplotlib.pyplot as plt


def run(
    arch,
    samples: int = 51,
    avg_frames: int = 1,
    plot: bool = False,
    verbose: bool = True,
    return_metadata: bool = False
):
    metadata = {}

    # Build phase vector
    phases = np.linspace(0, 2 * np.pi, samples)

    # All possible input combinations:
    input_combinations = [c for r in range(arch.n_inputs + 1) for c in combinations(range(arch.n_inputs), r)]

    total_scans = len(input_combinations) * len(arch.shifters)
    scan_counter = 0

    # Scan for each active input combination
    results = {}
    for active_inputs in input_combinations:
        if verbose:
            print(f"Scanning active inputs: {active_inputs}")

        # Turn off all inactive inputs
        phobos.Injection().off()
        phobos.Injection().set_balanced(active_inputs)

        # For each shifter, turn others off and scan its phase
        data = []
        for shifter_idx, shifter in enumerate(arch.shifters):
            scan_counter += 1
            if verbose:
                print(f"  [{scan_counter}/{total_scans}] Shifter {shifter.channel}")

            # Turn off all topas
            arch.turn_off(verbose=False)

            # Scan on the specific shifter
            fluxes = []
            for phi in phases:
                # Set phase on the specific shifter
                shifter.set_phase(float(phi))

                # Get output flux
                outs = phobos.Cred3().get_outputs(stack=avg_frames)
                fluxes.append(outs)

            # Convert to numpy array
            fluxes = np.array(fluxes)  # shape: (n_phases, n_outputs)
            fluxes = fluxes.T # Transpose to (n_outputs, n_phases)
            
            data.append(fluxes)
        results[active_inputs] = data

    if plot:
        figs = plot_results(arch, results)
        metadata['figures'] = figs

    if return_metadata:
        return results, metadata
    else:
        return results

def plot_results(arch, results: dict):
    figs = {}
    input_combinations = list(results.keys())

    for i in range(arch.n_inputs + 1):
        comb_i_inputs = [c for c in input_combinations if len(c) == i]
        n_rows = len(arch.shifters)
        n_cols = len(comb_i_inputs)

        if n_cols == 0:
            continue

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(max(1, n_cols) * 5, n_rows * 4),
            tight_layout=True,
            squeeze=False,  # Always 2D indexing: axs[s, c]
        )

        for c, active_inputs in enumerate(comb_i_inputs):
            for s in range(n_rows):
                y = np.asarray(results[active_inputs][s], dtype=float)  # (n_outputs, n_phases)
                x = np.linspace(0.0, 2.0 * np.pi, y.shape[1])

                ax = axs[s, c]
                for o in range(arch.n_outputs):
                    ax.plot(x, y[o], marker="o", markersize=3, linewidth=1.2, label=f"Output {o}")

                ax.set_title(f"Inputs: {active_inputs}, Shifter: {s}")
                ax.set_xlabel("Phase [rad]")
                ax.set_ylabel("Flux [ADU]")
                ax.grid(alpha=0.3)
                if s == 0 and c == 0:
                    ax.legend()

        figs[f"{i}_inputs"] = fig

    return figs