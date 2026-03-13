"""Utility to perform a systematic phase scan of an architecture's shifters.

This module provides a single function `systematic_scan` that follows the
same basic logic as :meth:`phobos.classes.photonic_chip.arch.Arch.characterize`
but returns a compact dictionary mapping active input combinations to the
measured phase scans for each shifter.

The returned structure has the form::

    {
        (i1, i2, ...): {
            shifter_channel: ndarray(shape=(n_outputs, n_phases)),
            ...
        },
        ...
    }

The function intentionally remains small and hardware-driven: it uses
`phobos.DM()` to enable inputs and `phobos.Cred3()` to read the outputs. It
expects an `arch` object compatible with the PHOBos `Arch` API (has
`shifters`, `n_inputs`, `turn_off()` / `set_phases()` helpers, ...).

All text and docstrings are written in English to follow project rules.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np

import phobos
from phobos.classes.injection import Injection


def systematic_scan(
    arch,
    phase_samples: int = 51,
    avg_frames: int = 5,
    phase_range: Tuple[float, float] = (0.0, 2 * np.pi),
    settle_time: float = 0.05,
    plot: bool = False,
    return_metadata: bool = False,
    verbose: bool = True,
) -> Dict[Tuple[int, ...], Dict[int, np.ndarray]]:
    """Perform a systematic phase scan over all shifters for each input combo.

    Parameters
    ----------
    arch : phobos.classes.photonic_chip.Arch
        Architecture instance (e.g. `Arch6`) that exposes `shifters`,
        """Utility to perform a systematic phase scan of an architecture's shifters.

        This module provides a single function `systematic_scan` that follows the
        same basic logic as :meth:`phobos.classes.photonic_chip.arch.Arch.characterize`
        but returns a compact dictionary mapping active input combinations to the
        measured phase scans for each shifter.

        The returned structure has the form::

            {
                (i1, i2, ...): {
                    shifter_channel: ndarray(shape=(n_outputs, n_phases)),
                    ...
                },
                ...
            }

        The function intentionally remains small and hardware-driven: it uses
        high-level injection control via the `Injection` singleton when manipulating
        DM injection segments and `phobos.Cred3()` to read the outputs. It expects an
        `arch` object compatible with the PHOBos `Arch` API (has `shifters`,
        `n_inputs`, `turn_off()` / `set_phases()` helpers, ...).

        All text and docstrings are written in English to follow project rules.
        """

        from __future__ import annotations

        import time
        from itertools import combinations
        from typing import Dict, List, Tuple

        import numpy as np

        import phobos
        from phobos.classes.injection import Injection


        def systematic_scan(
            arch,
            phase_samples: int = 51,
            avg_frames: int = 5,
            phase_range: Tuple[float, float] = (0.0, 2 * np.pi),
            settle_time: float = 0.05,
            plot: bool = False,
            return_metadata: bool = False,
            verbose: bool = True,
        ) -> Dict[Tuple[int, ...], Dict[int, np.ndarray]]:
            """Perform a systematic phase scan over all shifters for each input combo.

            Parameters
            ----------
            arch : phobos.classes.photonic_chip.Arch
                Architecture instance (e.g. `Arch6`) that exposes `shifters`,
                `n_inputs` and `turn_off()` or `set_phases()`.
            phase_samples : int, optional
                Number of equally spaced phase points between start and stop
                (inclusive). Default is 51.
            avg_frames : int, optional
                Number of camera frames to average for each phase point. Default 5.
            phase_range : tuple, optional
                Phase start and stop values (radians). Default (0, 2*pi).
            settle_time : float, optional
                Time in seconds to wait after setting a phase before acquiring.
                Default 0.05s.
            plot : bool, optional
                If True, generate a matplotlib figure for each input combination.
                Each figure has rows = shifters scanned and columns = output channels.
                Default False.
            return_metadata : bool, optional
                If True, return a second value containing metadata including the
                generated figures (only meaningful when `plot=True`). Default False.
            verbose : bool, optional
                Print progress information.

            Returns
            -------
            dict
                Mapping from tuple(active_inputs) -> { shifter_channel: flux_array }

                `flux_array` is a NumPy array with shape (n_outputs, n_phases). The
                order of outputs follows the camera API (phobos.Cred3.get_outputs).
            """

            # Build phase vector
            phases = np.linspace(phase_range[0], phase_range[1], phase_samples)

            # All possible input combinations: include empty (no active inputs),
            # then 1-input, 2-inputs, ..., all
            all_inputs = list(range(1, arch.n_inputs + 1))
            input_combinations: List[Tuple[int, ...]] = []

            # include the "no active input" case
            input_combinations.append(())

            # 1-input combos
            for i in all_inputs:
                input_combinations.append((i,))

            # 2..(n_inputs)-input combos
            for r in range(2, len(all_inputs) + 1):
                for combo in combinations(all_inputs, r):
                    input_combinations.append(tuple(combo))

            # Result container
            results: Dict[Tuple[int, ...], Dict[int, np.ndarray]] = {}
            figures: Dict[Tuple[int, ...], object] = {}

            camera = phobos.Cred3()
            dm = phobos.DM()
            inj = Injection()

            total_scans = len(input_combinations) * len(arch.shifters)
            scan_counter = 0

            for active_inputs in input_combinations:
                if verbose:
                    if active_inputs:
                        print(f"Scanning active inputs: {active_inputs}")
                    else:
                        print("Scanning with no active inputs (all inputs parked)")

                # Ensure inputs: use Injection singleton to park or set balanced
                # Injection expects 0-based channel numbers; arch input indices are
                # 1-based here, so convert when needed.
                try:
                    inj.off()
                except Exception:
                    # best-effort: fall back to DM.off
                    try:
                        dm.off()
                    except Exception:
                        pass

                if len(active_inputs) > 0:
                    # convert to 0-based channels
                    channels = [int(i) - 1 for i in active_inputs]
                    try:
                        inj.balanced(channels)
                    except Exception:
                        # If injection balanced is not available for some reason,
                        # attempt to call balanced without args (all channels)
                        try:
                            inj.balanced()
                        except Exception:
                            # last resort: try DM.flat as fallback
                            try:
                                dm.flat(list(active_inputs))
                            except Exception:
                                pass

                # Prepare container for this input combo
                combo_dict: Dict[int, np.ndarray] = {}

                # For each shifter, turn others off and scan its phase
                for sh_idx, sh in enumerate(arch.shifters):
                    scan_counter += 1
                    if verbose:
                        print(f"  [{scan_counter}/{total_scans}] Shifter {sh.channel}")

                    # Turn off / reset architecture shifters (keeps inputs configuration)
                    try:
                        arch.turn_off(verbose=False)
                    except Exception:
                        # Fallback: set all phases to zero if available
                        try:
                            arch.set_phases([0.0] * len(arch.shifters), verbose=False)
                        except Exception:
                            pass

                    # Scan this shifter across phases and record averaged outputs
                    fluxes = []
                    for phi in phases:
                        # Set phase on the specific shifter
                        sh.set_phase(float(phi))

                        # Allow system to settle before acquisition
                        if settle_time > 0:
                            time.sleep(settle_time)

                        # Acquire several frames and average
                        samples = []
                        for _ in range(max(1, int(avg_frames))):
                            outs = camera.get_outputs()
                            samples.append(np.asarray(outs))

                        mean_out = np.mean(samples, axis=0)
                        fluxes.append(mean_out)

                    fluxes = np.array(fluxes)  # shape: (n_phases, n_outputs)
                    # Transpose to (n_outputs, n_phases) to match requested format
                    fluxes = fluxes.T

                    combo_dict[sh.channel] = fluxes

                results[tuple(active_inputs)] = combo_dict

                # Optionally plot the results for this input combination
                if plot:
                    try:
                        import matplotlib.pyplot as plt
                    except Exception:
                        if verbose:
                            print("⚠️ matplotlib not available, skipping plotting")
                        figures[tuple(active_inputs)] = None
                    else:
                        # Determine layout: rows = shifters, cols = n_outputs
                        shifter_channels = [sh.channel for sh in arch.shifters]
                        n_rows = len(shifter_channels)
                        # infer n_outputs from one of the flux arrays
                        sample_sh = next(iter(combo_dict.values()))
                        n_outputs = sample_sh.shape[0]

                        # Global y-limits for better comparison
                        y_min = np.inf
                        y_max = -np.inf
                        for arr in combo_dict.values():
                            y_min = min(y_min, np.min(arr))
                            y_max = max(y_max, np.max(arr))

                        fig, axs = plt.subplots(n_rows, n_outputs,
                                                figsize=(4 * n_outputs, 2.5 * n_rows),
                                                squeeze=False)
                        fig.suptitle(f"Inputs {tuple(active_inputs)} - phase scans")

                        for i_row, ch in enumerate(shifter_channels):
                            flux_arr = combo_dict.get(ch)
                            if flux_arr is None:
                                # empty row: turn off axis
                                for j in range(n_outputs):
                                    axs[i_row][j].axis('off')
                                continue

                            for j_col in range(n_outputs):
                                ax = axs[i_row][j_col]
                                ax.plot(phases, flux_arr[j_col, :], marker='o', ls='-')
                                ax.set_xlabel('Phase (rad)')
                                if j_col == 0:
                                    ax.set_ylabel(f'Shifter {ch}\nFlux')
                                ax.set_title(f'Out {j_col}')
                                ax.grid(True, alpha=0.3)
                                ax.set_ylim(y_min, y_max)

                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        figures[tuple(active_inputs)] = fig

            # Restore safe state: turn off architecture shifters and set injection to
            # balanced positions using the Injection singleton.
            try:
                arch.turn_off(verbose=False)
            except Exception:
                pass
            try:
                inj.balanced()
            except Exception:
                # fallback to DM.flat() or DM.off()
                try:
                    dm.flat()
                except Exception:
                    try:
                        dm.off()
                    except Exception:
                        pass

            if return_metadata:
                metadata = {'figures': figures if plot else None, 'phases': phases}
                return results, metadata

            return results
                # Fallback: set all phases to zero if available
                try:
                    arch.set_phases([0.0] * len(arch.shifters), verbose=False)
                except Exception:
                    pass

            # Scan this shifter across phases and record averaged outputs
            fluxes = []
            for phi in phases:
                # Set phase on the specific shifter
                sh.set_phase(float(phi))

                # Allow system to settle before acquisition
                if settle_time > 0:
                    time.sleep(settle_time)

                # Acquire several frames and average
                samples = []
                for _ in range(max(1, int(avg_frames))):
                    outs = camera.get_outputs()
                    samples.append(np.asarray(outs))

                mean_out = np.mean(samples, axis=0)
                fluxes.append(mean_out)

            fluxes = np.array(fluxes)  # shape: (n_phases, n_outputs)
            # Transpose to (n_outputs, n_phases) to match requested format
            fluxes = fluxes.T

            combo_dict[sh.channel] = fluxes

        results[tuple(active_inputs)] = combo_dict

        # Optionally plot the results for this input combination
        if plot:
            try:
                import matplotlib.pyplot as plt
            except Exception:
                if verbose:
                    print("⚠️ matplotlib not available, skipping plotting")
                figures[tuple(active_inputs)] = None
            else:
                # Determine layout: rows = shifters, cols = n_outputs
                shifter_channels = [sh.channel for sh in arch.shifters]
                n_rows = len(shifter_channels)
                # infer n_outputs from one of the flux arrays
                sample_sh = next(iter(combo_dict.values()))
                n_outputs = sample_sh.shape[0]

                # Global y-limits for better comparison
                y_min = np.inf
                y_max = -np.inf
                for arr in combo_dict.values():
                    y_min = min(y_min, np.min(arr))
                    y_max = max(y_max, np.max(arr))

                fig, axs = plt.subplots(n_rows, n_outputs,
                                        figsize=(4 * n_outputs, 2.5 * n_rows),
                                        squeeze=False)
                fig.suptitle(f"Inputs {tuple(active_inputs)} - phase scans")

                for i_row, ch in enumerate(shifter_channels):
                    flux_arr = combo_dict.get(ch)
                    if flux_arr is None:
                        # empty row: turn off axis
                        for j in range(n_outputs):
                            axs[i_row][j].axis('off')
                        continue

                    for j_col in range(n_outputs):
                        ax = axs[i_row][j_col]
                        ax.plot(phases, flux_arr[j_col, :], marker='o', ls='-')
                        ax.set_xlabel('Phase (rad)')
                        if j_col == 0:
                            ax.set_ylabel(f'Shifter {ch}\nFlux')
                        ax.set_title(f'Out {j_col}')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(y_min, y_max)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                figures[tuple(active_inputs)] = fig

    # Restore safe state: turn off architecture shifters and set injection to
    # balanced positions using the Injection singleton.
    try:
        arch.turn_off(verbose=False)
    except Exception:
        pass
    try:
        inj.balanced()
    except Exception:
        # fallback to DM.flat() or DM.off()
        try:
            dm.flat()
        except Exception:
            try:
                dm.off()
            except Exception:
                pass

    if return_metadata:
        metadata = {'figures': figures if plot else None, 'phases': phases}
        return results, metadata

    return results
