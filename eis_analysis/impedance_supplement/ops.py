import numpy as np
from impedance.models.circuits.fitting import wrapCircuit


# %% Element/Param from model parsers
class ImpedanceFunc:
    """
    Callable wrapper around get_impedance.
    Binds a model and constants, then can be called with (f, *params).
    """

    def __init__(self, model: str, constants: dict | None = None):
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string.")

        self.model = model

        if not isinstance(constants, dict):
            constants = {}

        self.constants = constants

    def __call__(
        self, f: np.ndarray | None = None, *params: int | float | np.number
    ) -> np.ndarray:
        """
        Returns the impedance array for the given frequencies and parameters.

        Parameters
        ----------
        f : np.ndarray
            Array of frequencies.
        params : tuple
            Model parameters.
        Returns
        -------
        np.ndarray
            Array of complex impedance values.
        """
        if len(params) == 1 and isinstance(params[0], (list, tuple, np.ndarray)):
            params = tuple(params[0])

        if not params:
            raise ValueError("Params must not be empty.")

        if f is None:
            f = np.logspace(-3, 6, 1000)

        Z = np.array(np.hsplit(wrapCircuit(self.model, self.constants)(f, *params), 2)).T
        return Z[:, 0] + 1j * Z[:, 1]


def get_impedance(
    f: np.ndarray | None = None,
    *params: int | float | np.number,
    model: str = "",
    constants: dict | None = None,
) -> np.ndarray:
    """
    Returns the impedance array for the given frequencies and parameters.

    Parameters
    ----------
    f : np.ndarray
        Array of frequencies.
    params : tuple
        Model parameters.
    model : str
        Circuit model string.
    constants : dict
        Constants for the circuit model.

    Returns
    -------
    np.ndarray
        Array of complex impedance values.
    """
    return ImpedanceFunc(model, constants)(f, *params)
