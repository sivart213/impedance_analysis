# Suggested placement: in your test_complex_data.py or similar test file

import numpy as np
import pandas as pd
import pytest

from testing.generators import check_result_type_and_print
from testing.rc_ckt_sim import RCCircuit
from eis_analysis.z_system.system import ComplexSystem
from eis_analysis.z_system.convert import convert
from eis_analysis.z_system.complexer import Complexer


def test_valid_forms_aliasing(dummy_transforms):
    """
    Clarifying comment: tests that all aliases for smoothed_form are registered and work.
    """
    arr = np.arange(5)
    for alias in ["S", "ƒₛₘ", "sm", "smooth", "smoothed"]:
        result = getattr(dummy_transforms, dummy_transforms._valid_forms[alias])(arr)
        check_result_type_and_print(result, np.ndarray, f"alias {alias} for smoothed_form")
        assert np.allclose(result, arr, atol=1e-1)


@pytest.fixture
def rc_data():
    """Fixture providing frequency and impedance data from RCCircuit for ComplexSystem tests."""
    rc = RCCircuit()
    return {
        "frequency": rc.freq,
        "impedance": rc.Z,
        "impedance_noisy": rc.Z_noisy,
    }


@pytest.fixture
def simple_data():
    """Fixture providing simple frequency and impedance arrays for edge case testing."""
    freq = np.array([1e1, 1e2, 1e3, 1e4])
    z = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
    return {"frequency": freq, "impedance": z}


@pytest.fixture
def rc_system():
    """Fixture providing a ComplexSystem with trusted impedance data."""
    ckt = RCCircuit(true_values=[24, 1e9, 1e-11], noise=0.01)
    return ComplexSystem(data=ckt.Z_noisy, frequency=ckt.freq, area=25, thickness=500e-4)


@pytest.mark.parametrize(
    "property_name, description",
    [
        # Clarifying comment: tests each main ComplexSystem property that returns a Complexer
        ("impedance", "impedance property returns correct Complexer"),
        ("admittance", "admittance property returns correct Complexer"),
        ("capacitance", "capacitance property returns correct Complexer"),
        ("modulus", "modulus property returns correct Complexer"),
        ("permittivity", "permittivity property returns correct Complexer"),
        ("relative_permittivity", "relative_permittivity property returns correct Complexer"),
        (
            "relative_permittivity_corrected",
            "relative_permittivity_corrected property returns correct Complexer",
        ),
        ("conductivity", "conductivity property returns correct Complexer"),
        ("resistivity", "resistivity property returns correct Complexer"),
    ],
)
def test_complex_system_main_properties(rc_data, property_name, description):
    """
    Test ComplexSystem main properties for correct type and shape.

    Clarifying comments:
    - Each test checks that the property returns a Complexer with the correct shape.
    - Does not check numerical values, only interface and type.
    """
    cs = ComplexSystem(data=rc_data["impedance"], frequency=rc_data["frequency"])
    result = getattr(cs, property_name)
    check_result_type_and_print(result, Complexer, description)
    # Clarifying comment: ensure result is Complexer and shape matches input
    assert isinstance(result, Complexer)
    assert result.array.shape == rc_data["impedance"].shape


def test_complex_system_aliases(rc_data):
    """
    Clarifying comment: tests that aliases in ComplexSystem map to correct properties.
    """
    cs = ComplexSystem(data=rc_data["impedance"], frequency=rc_data["frequency"])
    # Test a few representative aliases

    # Helper to get the underlying array for comparison
    def get_array(val):
        return val.array

    # Clarifying comments before each assert
    # tests that alias 'z' returns a Complexer and matches impedance property
    assert np.all(get_array(cs["z"]) == get_array(cs.impedance))
    # tests that alias 'y' returns a Complexer and matches admittance property
    assert np.all(get_array(cs["y"]) == get_array(cs.admittance))
    # tests that alias 'c' returns a Complexer and matches capacitance property
    assert np.all(get_array(cs["c"]) == get_array(cs.capacitance))
    # tests that alias 'm' returns a Complexer and matches modulus property
    assert np.all(get_array(cs["m"]) == get_array(cs.modulus))
    # tests that alias 'e' returns a Complexer and matches permittivity property
    assert np.all(get_array(cs["e"]) == get_array(cs.permittivity))
    # tests that alias 'e_r' returns a Complexer and matches relative_permittivity property
    assert np.all(get_array(cs["e_r"]) == get_array(cs.relative_permittivity))
    # tests that alias 'sigma' returns a Complexer and matches conductivity property
    assert np.all(get_array(cs["sigma"]) == get_array(cs.conductivity))
    # tests that alias 'rho' returns a Complexer and matches resistivity property
    assert np.all(get_array(cs["rho"]) == get_array(cs.resistivity))


def test_complex_system_get_df(rc_data):
    """
    Clarifying comment: tests get_df returns DataFrame with correct columns and attrs.
    """
    cs = ComplexSystem(data=rc_data["impedance"], frequency=rc_data["frequency"])
    result = cs.get_df("impedance", "admittance")
    check_result_type_and_print(result, pd.DataFrame, "get_df returns DataFrame")
    # Clarifying comment: check that DataFrame columns are as expected
    expected_cols = ["impedance.real", "impedance.imag", "admittance.real", "admittance.imag"]
    for col in expected_cols:
        assert col in result.columns
    # Clarifying comment: check that attrs contains area and thickness
    assert "area" in result.attrs
    assert "thickness" in result.attrs


@pytest.mark.parametrize(
    "from_form",
    [
        "impedance",
        "admittance",
        "modulus",
        "capacitance",
        "resistivity",
        "conductivity",
        "permittivity",
        "relative_permittivity",
        "relative_permittivity_corrected",
        "susceptibility",
    ],
)
def test_convert_all_forms(rc_system, from_form):
    """
    Test convert function for all supported forms.
    Clarifying comment: Each form is converted back to impedance and compared to the trusted source.
    """
    data = rc_system.get_complexer(from_form)
    kwargs = {}
    # Clarifying comment: pass correct phys_const for special forms
    if from_form == "relative_permittivity_corrected":
        kwargs["phys_const"] = rc_system.val_towards_zero(rc_system.conductivity.real)
    elif from_form == "susceptibility":
        kwargs["phys_const"] = rc_system.val_towards_infinity(rc_system.relative_permittivity.real)
    # Use convert to get impedance
    result = convert(data, from_form=from_form, to_form="impedance", system=rc_system, **kwargs)
    np.testing.assert_allclose(
        result.array,
        rc_system.impedance.array,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"Conversion from {from_form} to impedance failed when passing Complexer",
    )

    # Also test with .array input
    result = convert(
        data.array, from_form=from_form, to_form="impedance", system=rc_system, **kwargs
    )
    np.testing.assert_allclose(
        result.array,
        rc_system.impedance.array,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"Conversion from {from_form} to impedance failed when passing ndarray",
    )


def test_complex_system_update_precedence(rc_system):
    """
    Clarifying comment: Tests precedence of area/thickness in update when data is passed as:
    - raw array (should use instance or explicit kwargs)
    - Complexer (should use instance or explicit kwargs)
    - ComplexSystem (should use area/thickness from the passed system)
    """
    freq = rc_system.frequency
    ones = np.ones_like(freq, dtype=complex)

    # Case 1: Update with raw array, explicit area/thickness
    cs1 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    cs1.update(
        rc_system.e_r.array,
        form="relative_permittivity",
        area=rc_system.area,
        thickness=rc_system.thickness,
    )
    print("Case 1: raw array, explicit area/thickness")
    np.testing.assert_allclose(cs1.impedance.array, rc_system.array, rtol=1e-6, atol=1e-8)

    # Case 2: Update with Complexer, explicit area/thickness
    cs2 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    cs2.update(
        rc_system.e_r,
        form="relative_permittivity",
        area=rc_system.area,
        thickness=rc_system.thickness,
    )
    print("Case 2: Complexer, explicit area/thickness")
    np.testing.assert_allclose(cs2.impedance.array, rc_system.array, rtol=1e-6, atol=1e-8)

    # Case 3: Update with dataframe (should use rc_system area/thickness)
    cs3 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    rel_df = rc_system.get_df("relative_permittivity")
    cs3.update(rel_df, form="relative_permittivity", area=1.5, thickness=1.5)
    print("Case 3: DataFrame, area/thickness from data")
    np.testing.assert_allclose(cs3.array, rc_system.array, rtol=1e-6, atol=1e-8)

    # Case 4: Update with another ComplexSystem (should use its area/thickness)
    cs4 = ComplexSystem(data=ones, frequency=freq, area=1.0, thickness=1.0)
    rel_perm_system = ComplexSystem(rc_system.e_r.array, freq, rc_system.thickness, rc_system.area)
    cs4.update(rel_perm_system, form="relative_permittivity", area=1.5, thickness=1.5)
    print("Case 4: ComplexSystem, area/thickness from data")
    np.testing.assert_allclose(cs4.impedance.array, rc_system.array, rtol=1e-6, atol=1e-8)


def test_complex_system_update_and_ordering(simple_data):
    """
    Tests ComplexSystem.update and ensure_order integration:
    - Order is preserved when both data and frequency are provided.
    - Order is enforced when reversed frequency/data are given.
    - expected_z_at_dc is only applied for impedance form.
    - Malformed frequency arrays are handled gracefully.
    """

    freq = simple_data["frequency"]
    z = simple_data["impedance"]
    freq_rev = freq[::-1]
    z_rev = z[::-1]

    # --- Case 1: Impedance update with reversed input ---
    cs = ComplexSystem(data=z, frequency=freq, order="asc")
    cs.update(data=z_rev, frequency=freq_rev, expected_z_at_dc=True, form="impedance")

    # Frequency should be sorted ascending
    assert np.all(np.diff(cs.frequency) > 0)
    # Data should either remain reversed or be inverted depending on DC expectation
    assert np.allclose(cs.impedance.array, z_rev)
    cs.order = "desc"
    assert np.all(np.diff(cs.frequency) < 0)
    assert np.allclose(cs.impedance.array, z)

    # --- Case 3: Malformed frequency array (not monotonic enough) ---
    malformed_freq = freq.copy()
    malformed_freq[1] = 1  # Break strict monotonicity
    # z_data = np.linspace(1, 5, len(malformed_freq))
    cs.order = "asc"
    cs.update(
        data=z_rev,
        frequency=malformed_freq,
        expected_z_at_dc=True,
        form="impedance",
        tolerance=0.1,
    )
    # Should not raise, but frequency may be returned unchanged if monotonicity < 0.9

    assert np.all(np.diff(cs.frequency) > 0)
    assert not np.allclose(cs.frequency, malformed_freq)

    idxs = np.argsort(malformed_freq)
    assert not np.allclose(cs.array, z_rev[idxs])

    cs.update(
        data=z_rev,
        frequency=malformed_freq,
        expected_z_at_dc=True,
        form="impedance",
        tolerance=0.3,
    )
    assert np.all(np.diff(cs.frequency) > 0)
    assert not np.allclose(cs.frequency, malformed_freq)

    idxs = np.argsort(malformed_freq)
    assert np.allclose(cs.array, z_rev[idxs])
    # assert len(cs.impedance.array) == len(z_data)


# def test_complex_system_update_and_ordering(simple_data):
#     """
#     Clarifying comment: tests update method and frequency/data ordering logic.
#     """
#     cs = ComplexSystem(data=simple_data["impedance"], frequency=simple_data["frequency"])
#     # Reverse frequency and update
#     freq_rev = simple_data["frequency"][::-1]
#     z_rev = simple_data["impedance"][::-1]
#     cs.update(data=z_rev, frequency=freq_rev)
#     # Clarifying comment: when both data and frequency are provided, order is preserved
#     assert np.all(cs.frequency == freq_rev)
#     assert np.all(cs.impedance.array == z_rev)

#     # Clarifying comment: when only frequency is updated, order is enforced
#     cs2 = ComplexSystem(data=simple_data["impedance"], frequency=simple_data["frequency"])
#     # cs2.update(frequency=freq_rev)
#     # Frequency should now be sorted ascending
#     assert np.all(np.diff(cs2.frequency) > 0)
#     # Impedance should be reordered to match sorted frequency
#     # sorted_indices = np.argsort(freq_rev)
#     # assert np.all(cs2.impedance.array == simple_data["impedance"][sorted_indices])
#     assert np.all(cs.impedance.array == z_rev)


@pytest.mark.parametrize(
    "key, expected, description",
    [
        # Clarifying comment: test valid keys and some invalid ones
        ("impedance", True, "impedance is a valid key"),
        ("z", True, "z is a valid alias"),
        ("foo", False, "foo is not a valid key"),
        ("sigma", True, "sigma is a valid alias"),
        ("relative_permittivity", True, "relative_permittivity is a valid key"),
        ("not_a_property", False, "not_a_property is not a valid key"),
    ],
)
def test_complex_system_is_valid_key(key, expected, description):
    """
    Clarifying comment: tests is_valid_key static method for various keys.
    """
    result = ComplexSystem.is_valid_key(key, dummy_test=False)
    check_result_type_and_print(result, bool, description)
    assert result is expected


def test_complex_system_repr(rc_data):
    """
    Clarifying comment: tests __repr__ returns expected string.
    """
    cs = ComplexSystem(data=rc_data["impedance"], frequency=rc_data["frequency"])
    result = repr(cs)
    check_result_type_and_print(result, str, "repr returns string")
    assert result.startswith("ComplexSystem(")
