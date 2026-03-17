import numpy as np
from IPython.display import Math, display


def _format_complex_polar(value: complex) -> str:
    """Format a complex number as z e^{a i pi}.

    Parameters
    ----------
    value : complex
        Complex value to format.

    Returns
    -------
    str
        LaTeX-ready string with z formatted as ``:.2g`` and a as ``:.2f``.
    """
    z = np.abs(value)
    a = np.angle(value) / np.pi
    return f"{z:.2g}e^{{{a:.2f}i\\pi}}"


def display_complex_matrix(matrix, name="M"):
    """Display a complex matrix in a Jupyter notebook using LaTeX.

    Parameters
    ----------
    matrix : array-like
        Input matrix, converted to a 2D NumPy array.
    name : str, optional
        Matrix symbol used in the LaTeX expression, by default "M".
    """
    arr = np.asarray(matrix, dtype=complex)

    if arr.ndim != 2:
        raise ValueError("display_complex_matrix expects a 2D array.")

    rows = []
    for row in arr:
        rows.append(" & ".join(_format_complex_polar(v) for v in row))

    body = r" \\ ".join(rows)
    latex = rf"{name}=\begin{{pmatrix}}{body}\end{{pmatrix}}"
    display(Math(latex))


def display_complex_vector(matrix, name="v"):
    """Display a complex vector in a Jupyter notebook using LaTeX.

    Parameters
    ----------
    matrix : array-like
        Input vector. Accepted shapes are (n,), (n, 1), or (1, n).
        Displayed as a column vector.
    name : str, optional
        Vector symbol used in the LaTeX expression, by default "v".
    """
    arr = np.asarray(matrix, dtype=complex)

    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2 and 1 in arr.shape:
        vec = arr.reshape(-1)
    else:
        raise ValueError("display_complex_vector expects a 1D array or a single-row/single-column 2D array.")

    body = r" \\ ".join(_format_complex_polar(v) for v in vec)
    latex = rf"{name}=\begin{{pmatrix}}{body}\end{{pmatrix}}"
    display(Math(latex))