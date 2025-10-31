# -*- coding: utf-8 -*-
"""
Insert module description/summary.

Provide any or all of the following:
1. extended summary
2. routine listings/functions/classes
3. see also
4. notes
5. references
6. examples

@author: j2cle
Created on Thu Mar 27 17:01:27 2025
"""


import pickle

import cloudpickle


def check_picklability(obj, cloud=False, auto_print=False):
    """
    Test each attribute of an object to see if it can be pickled.
    Returns a dict mapping attribute names to True/False.
    """
    results = {}
    for name, value in vars(obj).items():
        try:
            data = cloudpickle.dumps(value) if cloud else pickle.dumps(value)
            cloudpickle.loads(data) if cloud else pickle.loads(data)
            results[name] = True
        except Exception as e:
            if hasattr(value, "__len__") and len(value) > 10:
                results[name] = f"Unpicklable: {type(value).__name__} ({e})"
            else:
                results[name] = f"Unpicklable: {type(value).__name__} w/ value: {value} ({e})"
    if auto_print:
        for name, result in results.items():
            if result is not True:
                print(f"{name} -> {result}")
    return results


def check_attr_picklability(obj, cloud=False, auto_print=False):
    """
    Test each attribute of an object to see if it can be pickled.
    Returns a dict mapping attribute names to True/False.
    """
    results = {}
    for name in dir(obj):
        if name.startswith("__"):
            continue
        value = "<initialized state>"
        try:
            value = getattr(obj, name)
            data = cloudpickle.dumps(value) if cloud else pickle.dumps(value)
            cloudpickle.loads(data) if cloud else pickle.loads(data)
            results[name] = True
        except Exception as e:
            if value == "<initialized state>":
                results[name] = f"Failed to retrieve attr {name} ({e})"
                continue
            if hasattr(value, "__len__") and len(value) > 10:
                results[name] = f"Unpicklable: {type(value).__name__} ({e})"
            else:
                results[name] = f"Unpicklable: {type(value).__name__} w/ value: {value} ({e})"
    if auto_print:
        for name, result in results.items():
            if result is not True:
                print(f"{name} -> {result}")
    return results


def check_deep_picklability(
    obj, _path="root", _seen=None, cloud=False, eval_dir=True, auto_print=False
):
    """
    Recursively test picklability of an object and its contents.
    Returns a dict mapping path -> error message for unpicklable components.
    """

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return {}, {}
    _seen.add(oid)

    # errors = {}
    # objects = {}

    # Try pickling the object itself
    try:
        data = cloudpickle.dumps(obj) if cloud else pickle.dumps(obj)
        restored = cloudpickle.loads(data) if cloud else pickle.loads(data)
        repr(restored)
        errors = {_path + f" ({type(obj).__name__})": f"Picklable: {restored}"}
        objects = {_path + f" ({type(obj).__name__})": restored}
        if len(_seen) > 1:
            return errors, objects
    except Exception as e:
        # errors[_path + f" ({type(obj).__name__})"] = str(e)
        errors = objects = {_path + f" ({type(obj).__name__})": str(e)}

    # If it's a container, dive deeper
    if isinstance(obj, dict):
        for k, v in obj.items():
            e_res, o_res = check_deep_picklability(v, f"{_path}[{k!r}]", _seen, cloud, eval_dir)
            errors |= e_res
            objects |= o_res
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            e_res, o_res = check_deep_picklability(v, f"{_path}[{i}]", _seen, cloud, eval_dir)
            errors |= e_res
            objects |= o_res
    # elif isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type(None))):
    #     errors[_path + f" ({type(obj).__name__})"] = obj
    else:
        items = []
        if hasattr(obj, "__dict__"):
            items = list(obj.__dict__.keys())
        if eval_dir:
            items += [k for k in dir(obj) if not k.startswith("__") and k not in items]
        for k in items:
            try:
                val = getattr(obj, k)
            except Exception as e:
                errors[f"{_path}.{k}"] = f"Attribute access error: {e}"
                continue
            e_res, o_res = check_deep_picklability(val, f"{_path}.{k}", _seen, cloud, eval_dir)
            errors |= e_res
            objects |= o_res

    if auto_print and errors:
        for path, err in errors.items():
            print(f"{path} -> {err}")

    return errors, objects


# def check_deep_picklability(obj, _path="root", _seen=None, cloud=False, eval_dir=True, auto_print=False):
#     """
#     Recursively test picklability of an object and its contents.
#     Returns a list of (path, error) for unpicklable components.
#     """

#     if _seen is None:
#         _seen = set()
#     oid = id(obj)
#     if oid in _seen:
#         return []
#     _seen.add(oid)

#     # Try pickling the object itself
#     try:
#         cloudpickle.dumps(obj) if cloud else pickle.dumps(obj)
#         return []
#     except Exception as e:
#         errors = [(f"{_path} ({type(obj).__name__})", str(e))]

#     # If it's a container, dive deeper
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             errors.extend(check_deep_picklability(v, f"{_path}[{k!r}]", _seen, cloud, eval_dir))
#     elif isinstance(obj, (list, tuple, set, frozenset)):
#         for i, v in enumerate(obj):
#             errors.extend(check_deep_picklability(v, f"{_path}[{i}]", _seen, cloud, eval_dir))
#     else:
#         items = []
#         if hasattr(obj, "__dict__"):
#             items = list(obj.__dict__.keys())
#         items += [
#             k for k in dir(obj) if not k.startswith("__") and k not in items and hasattr(obj, k)
#         ]
#         for k in items:
#             errors.extend(
#                 check_deep_picklability(getattr(obj, k), f"{_path}.{k}", _seen, cloud, eval_dir)
#             )

#     if auto_print and errors:
#         for path, err in errors:
#             print(f"{path} -> {err}")
#     return errors
