# Suggested placement: in your test_complex_data.py or similar test file


import numpy as np
import pytest

from testing.rc_ckt_sim import RCCircuit
from eis_analysis.z_system.system import ComplexSystem
from eis_analysis.z_system.imped_parsing import ItemTransforms


# ---- Fixtures ----
@pytest.fixture
def simple_data():
    """Fixture providing simple frequency and impedance arrays for edge case testing."""
    freq = np.array([1e1, 1e2, 1e3, 1e4])
    z = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
    return {"frequency": freq, "impedance": z}


@pytest.fixture(scope="module")
def rc_ckt():
    """Fixture providing frequency and impedance data from RCCircuit."""
    return RCCircuit(
        freq=(-4, 7, 100), true_values=[24, 1e9, 1e-11], noise=0.001, lock_instance=True
    )


@pytest.fixture
def rc_data(rc_ckt):
    """Fixture providing frequency and impedance data from RCCircuit for ComplexSystem tests."""
    return {
        "frequency": rc_ckt.freq,
        "impedance": rc_ckt.Z,
        "impedance_noisy": rc_ckt.Z_noisy,
    }


@pytest.fixture(scope="module")
def rc_system(rc_ckt):
    """Fixture providing a ComplexSystem with trusted impedance data."""
    # ckt = RCCircuit(true_values=[24, 1e9, 1e-11], noise=0.01)
    return ComplexSystem(
        data=rc_ckt.Z_noisy, frequency=rc_ckt.freq, area=25, thickness=500e-4, writeable=False
    )


@pytest.fixture
def complexer_test_arrays():
    """Fixture providing test arrays for Complexer tests."""
    return {
        "single": np.array([1 + 2j]),
        "multi": np.array([1 + 2j, 3 - 4j, -1 + 0.5j]),
        "real_only": np.array([1.0, 2.0, 3.0]),
        "imag_only": np.array([1j, -2j, 3j]),
        "2d_cartesian": np.array([[1, 2], [3, -4], [-1, 0.5]]),  # (real, imag)
    }


# class DummyParser(BaseParser, form_str="_form"):
#     def __init__(self):
#         self.value = np.array([1, 2, 3])

#     def square_form(self, x):
#         return np.array(x) ** 2

#     def cube_form(self, x):
#         return np.array(x) ** 3


# @pytest.fixture
# def dummy_parser():
#     return DummyParser()


class DummyTransforms(ItemTransforms):
    def __init__(self):
        super().__init__()
        self.default_x = "dummy_arr"
        self.dummy_arr = np.linspace(0, 1, 5)


@pytest.fixture
def dummy_transforms():
    return DummyTransforms()
