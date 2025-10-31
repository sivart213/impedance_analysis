import pytest
import numpy as np
import pandas as pd

from eis_analysis.dict_ops import flat_dict_to_df, dict_to_df
# from ..dict_ops import flat_dict_to_df, dict_to_df
# test/test_flat_dict_to_df.py



def test_flat_dict_to_df_basic():
    d = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = flat_dict_to_df(d)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (3, 2)
    np.testing.assert_array_equal(df["a"], [1, 2, 3])
    np.testing.assert_array_equal(df["b"], [4, 5, 6])

def test_flat_dict_to_df_mixed_lengths():
    d = {"a": [1, 2, 3], "b": [4, 5]}
    df = flat_dict_to_df(d)
    # Should pick the most common length (3)
    assert isinstance(df, pd.DataFrame)
    assert "a" in df.columns
    assert df.shape[0] == 3

def test_flat_dict_to_df_scalars_and_lists():
    d = {"a": [1, 2, 3], "b": 42, "c": "foo"}
    df = flat_dict_to_df(d)
    assert isinstance(df, pd.DataFrame)
    assert "a" in df.columns
    assert df.attrs["b"] == 42
    assert df.attrs["c"] == "foo"

def test_flat_dict_to_df_single_element_lists():
    d = {"a": [1, 2, 3], "b": [99]}
    df = flat_dict_to_df(d)
    assert isinstance(df, pd.DataFrame)
    assert "a" in df.columns
    assert df.attrs["b"] == 99

def test_flat_dict_to_df_nested_dicts():
    d = {"a": [1, 2, 3], "meta": {"foo": "bar", "baz": 7}}
    df = flat_dict_to_df(d)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs["foo"] == "bar"
    assert df.attrs["baz"] == 7

def test_flat_dict_to_df_with_dataframe():
    d = {"a": [1, 2, 3]}
    df0 = pd.DataFrame({"x": [10, 20, 30]})
    d["df"] = df0
    df = flat_dict_to_df(d)
    assert isinstance(df, pd.DataFrame)
    assert "x" in df.columns or "a" in df.columns

def test_flat_dict_to_df_with_series():
    d = {"a": [1, 2, 3], "s": pd.Series([7, 8, 9], index=["x", "y", "z"])}
    df = flat_dict_to_df(d)
    assert isinstance(df, pd.DataFrame)
    # Series should be flattened into attrs or columns
    assert "a" in df.columns

def test_dict_to_df_equivalence():
    d = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df1 = flat_dict_to_df(d)
    df2 = dict_to_df(d)
    assert isinstance(df2, pd.DataFrame)
    pd.testing.assert_frame_equal(df1, df2)

# Generate some test data for quick manual comparison
def test_generate_example_data():
    d1 = {"x": np.arange(5), "y": np.linspace(0, 1, 5), "meta": {"unit": "V"}}
    d2 = {"foo": [1], "bar": [2, 3, 4], "baz": "hello"}
    d3 = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    print("d1:", d1)
    print("d2:", d2)
    print("d3:", d3)
    df1 = flat_dict_to_df(d1)
    df2 = flat_dict_to_df(d2)
    df3 = flat_dict_to_df(d3)
    print("df1:\n", df1)
    print("df2:\n", df2)
    print("df3:\n", df3)