import pytest

from testing.generators import (
    generate_flat_dict,
    check_result_type_and_print,
)

from ..dict_manipulators import (
    nest_dict,
    check_dict,
    filter_dict,
    update_dict,
    flatten_dict,
    safe_deepcopy,
    separate_dict,
    dict_level_ops,
    flip_dict_levels,
    push_non_dict_items,
    truncate_dict_levels,
    merge_unique_sub_dicts,
)


# %% Fixtures
@pytest.fixture
def flat_dict_factory():
    """
    Factory fixture to create a flat dict with customizable separator, depth, number of items, and values.
    Passes through all keyword arguments to generate_key_list and generate_value_list.
    """
    return generate_flat_dict


# %% Tests for flatten_dict
@pytest.mark.parametrize(
    "sep, levels, num_items, description",
    [
        ("/", 2, 0, "empty dict"),
        ("/", 1, 3, "1-level, string keys"),
        ("/", 2, 5, "2-level, string keys"),
        (".", 3, 6, "3-level, string keys, dot sep"),
        (None, 2, 4, "2-level, tuple keys"),
        ("__DEFAULT__", 2, 4, "default separator (should use '/')"),
        ("__LIST__", 2, 4, "input as list of key-value pairs"),
    ],
)
def test_flatten_and_nest_dict_roundtrip(flat_dict_factory, sep, levels, num_items, description):
    kwargs = {"sep": sep}
    result_type = dict
    if sep == "__DEFAULT__":
        kwargs = {}
        sep = "/"
    elif sep == "__LIST__":
        sep = "/"
        result_type = list

    flat = flat_dict_factory(sep=sep, levels=levels, num_items=num_items)
    if result_type is list:
        flat = list(flat.values())

    nested = nest_dict(flat, **kwargs)  # type: ignore[call-arg]
    check_result_type_and_print(nested, result_type, f"{description} - nested")
    roundtrip = flatten_dict(nested, **kwargs)
    check_result_type_and_print(roundtrip, result_type, f"{description} - roundtrip")

    assert roundtrip == flat, f"Failed roundtrip for: {description}"


@pytest.mark.parametrize(
    "input_dict, dtype, description",
    [
        ({"a": 1, "b": {"c": 2}}, dict, "simple nested dict"),
        ([1, 2, 3], list, "list input"),
        ({"x": [1, 2], "y": {"z": [3, 4]}}, dict, "dict with lists"),
    ],
)
def test_safe_deepcopy(input_dict, dtype, description):
    copy = safe_deepcopy(input_dict)
    check_result_type_and_print(copy, dtype, description)
    assert copy == input_dict
    assert copy is not input_dict


@pytest.mark.parametrize(
    "base, update, expected, description",
    [
        (
            {"a": 1, "b": {"c": 2}},
            {"b": {"d": 3}},
            {"a": 1, "b": {"c": 2, "d": 3}},
            "update nested",
        ),
        ({"x": 1}, {"x": 2, "y": 3}, {"x": 2, "y": 3}, "update flat"),
    ],
)
def test_update_dict(base, update, expected, description):
    base_copy = safe_deepcopy(base)
    check_result_type_and_print(base_copy, dict, description)
    update_dict(base_copy, update)
    assert base_copy == expected


@pytest.mark.parametrize(
    "base, filt, expected, description",
    [
        ({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": None}}, {"b": {"c": 2}}, "filter nested"),
        ({"x": 1, "y": 2}, {"x": None}, {"x": 1}, "filter flat"),
    ],
)
def test_filter_dict(base, filt, expected, description):
    result = filter_dict(base, filt)
    check_result_type_and_print(result, dict, description)
    assert result == expected


@pytest.mark.parametrize(
    "to_check, base, expected, description",
    [
        ({"c": 1}, {"a": {"b": {"c": 2}}}, {"a": {"b": {"c": 1}}}, "nest to_check"),
        ({"a": 1}, {"a": 2, "b": 3}, {"a": 1}, "key at top level"),
        ({}, {"a": 1}, {}, "empty to_check"),
    ],
)
def test_check_dict(to_check, base, expected, description):
    result = check_dict(to_check, base)
    check_result_type_and_print(result, dict, description)
    assert result == expected


@pytest.mark.parametrize(
    "data, op, level, expected, description",
    [
        (
            {"a": {"b": 2}},
            lambda d: {k: v * 2 for k, v in d.items()} if isinstance(d, dict) else d,
            2,
            {"a": {"b": 4}},
            "double at level 2",
        ),
        (
            {"x": 1, "y": 2},
            lambda d: {k: v + 1 for k, v in d.items()},
            1,
            {"x": 2, "y": 3},
            "increment at level 1",
        ),
    ],
)
def test_dict_level_ops(data, op, level, expected, description):
    result = dict_level_ops(data, op, level)
    check_result_type_and_print(result, dict, description)
    assert result == expected


@pytest.mark.parametrize(
    "levels, max_levels, sep, merge_at, description",
    [
        (3, 2, "/", "outer", "truncate 3->2 outer"),
        (3, 2, "/", "inner", "truncate 3->2 inner"),
        (2, 1, None, "outer", "truncate 2->1 tuple keys"),
    ],
)
def test_truncate_dict_levels(levels, max_levels, sep, merge_at, description):
    result = generate_flat_dict(levels=levels, num_items=8, sep=sep)
    check_result_type_and_print(result, dict, description)
    truncated = truncate_dict_levels(result, max_levels=max_levels, sep=sep, merge_at=merge_at)

    # Should not exceed max_levels of nesting
    def depth(obj):
        if not isinstance(obj, dict) or not obj:
            return 0
        return 1 + max(depth(v) for v in obj.values())

    assert (
        depth(truncated) <= max_levels
    ), f"{description}: depth {depth(truncated)} > {max_levels}"


@pytest.mark.parametrize(
    "levels, description",
    [
        (2, "flip 2-level dict"),
        (3, "flip 3-level dict"),
        (0, "flip all levels"),
    ],
)
def test_flip_dict_levels(levels, description):
    flat_data = generate_flat_dict(levels=max(2, levels), num_items=6)
    data = nest_dict(flat_data)
    result = flip_dict_levels(data, levels=levels)
    check_result_type_and_print(result, dict, description)
    assert result != data, f"{description}: flip did not change dict"
    # Flipping twice should return to original
    double_flipped = flip_dict_levels(result, levels=levels)
    assert double_flipped == data, f"{description}: double flip did not restore original"


@pytest.mark.parametrize(
    "input_dict, expected, description",
    [
        ({"a": {"x": 1}, "b": 2}, {"a": {"x": 1, "b": 2}}, "push non-dict to subdict"),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, "all non-dict"),
        (
            {"a": {"x": 1, "y": 2}, "b": {"z": 3}, "c": 4},
            {"a": {"x": 1, "y": 2, "c": 4}, "b": {"z": 3, "c": 4}},
            "multiple subdicts",
        ),
    ],
)
def test_push_non_dict_items(input_dict, expected, description):
    result = push_non_dict_items(input_dict)
    check_result_type_and_print(result, dict, description)
    assert result == expected


@pytest.mark.parametrize(
    "data, keep_keys, expected, description",
    [
        ({"a": {"x": 1}, "b": {"y": 2}}, None, {"x": 1, "y": 2}, "merge unique subdicts"),
        ({"a": {"x": 1}, "b": {"x": 2}}, None, {"a": {"x": 1}, "b": {"x": 2}}, "not unique keys"),
        ({"a": {"x": 1}, "b": 2}, None, {"a": {"x": 1}, "b": 2}, "mixed dict and non-dict"),
    ],
)
def test_merge_unique_sub_dicts(data, keep_keys, expected, description):
    result = merge_unique_sub_dicts(data, keep_keys)
    check_result_type_and_print(result, dict, description)
    assert result == expected


@pytest.mark.parametrize(
    "data, search_terms, reject_terms, expected_keys, description",
    [
        (
            {"foo1": 1, "bar2": 2, "baz3": 3},
            ["foo", "bar"],
            None,
            ["foo", "bar", "residuals"],
            "basic search",
        ),
        (
            {"alpha": 1, "beta": 2, "gamma": 3},
            ["alpha"],
            ["gamma"],
            ["alpha", "residuals"],
            "reject term",
        ),
    ],
)
def test_separate_dict(data, search_terms, reject_terms, expected_keys, description):
    result = separate_dict(data, search_terms, reject_terms)
    check_result_type_and_print(result, dict, description)
    assert set(result.keys()) == set(expected_keys)
