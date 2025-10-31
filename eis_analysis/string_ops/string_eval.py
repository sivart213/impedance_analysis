# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re
import ast
import operator as op
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from difflib import SequenceMatcher
from collections.abc import Callable, Hashable

import numpy as np

T = TypeVar("T")  # The return type of parse()


class AbstractEval(ABC):
    @abstractmethod
    def eval(self, node: Any) -> Any:
        pass

    @abstractmethod
    def parse(self, expr: Any) -> Any:
        pass

    # raise NotImplementedError("Subclasses must implement the eval method")


class ASTEvaluatorBase(Generic[T]):
    def __init__(self, error_mode: int | str = "str", **_):
        if isinstance(error_mode, int) and not isinstance(error_mode, bool):
            if error_mode == 0:
                self.error_mode = "str"
            elif error_mode == 1:
                self.error_mode = "raise"
            else:
                raise ValueError("error_mode must be 0, or 1 if an integer")
        elif str(error_mode).lower() in ("raise", "str"):
            self.error_mode = str(error_mode).lower()
        else:
            for opt in ("raise", "str"):
                if opt in str(error_mode).lower():
                    self.error_mode = opt
                    break
            else:
                raise ValueError("error_mode must be 'raise' or 'str' if a string")

        self.source = ""

    def eval(self, node: ast.AST) -> Any:
        try:
            method = getattr(self, f"eval_{type(node).__name__}", None)
            if method is None:
                raise ValueError(f"Unsupported node type: {type(node).__name__}")
            return method(node)
        except Exception as e:
            if self.error_mode == "raise":
                raise e
            return self._get_source_segment(node)

    def parse(self, expr: Any) -> str | T:
        if not isinstance(expr, str):
            return expr
        self.source = expr
        try:
            tree = ast.parse(expr, mode="eval")
            return self.eval(tree.body)
        except SyntaxError as e:
            if self.error_mode == "raise":
                raise e
            return expr

    def _get_source_segment(self, node: ast.AST) -> str:
        """Return the exact source text for a node."""
        try:
            return ast.get_source_segment(self.source, node) or "<unknown>"
        except AttributeError:
            return "<unknown>"


class MathEvalMixin(AbstractEval):
    OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def eval_Name(self, node: ast.Name) -> Any:
        if node.id in ("e", "pi", "inf", "nan"):
            return getattr(np, node.id)
        raise ValueError(f"Name '{node.id}' is not allowed")

    def eval_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")

    def eval_BinOp(self, node: ast.BinOp) -> Any:
        return self.OPERATORS[type(node.op)](self.eval(node.left), self.eval(node.right))

    def eval_UnaryOp(self, node: ast.UnaryOp):
        return self.OPERATORS[type(node.op)](self.eval(node.operand))


class ContainerEvalMixin(AbstractEval):
    def _is_hash(self, obj: Any) -> Hashable:
        try:
            hash(obj)
            return obj
        except TypeError:
            return str(obj)

    def eval_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def eval_Name(self, node: ast.Name) -> Any:
        return node.id

    def eval_List(self, node: ast.List):
        return [self.eval(elt) for elt in node.elts]

    def eval_Tuple(self, node: ast.Tuple):
        return tuple(self.eval(elt) for elt in node.elts)

    def eval_Set(self, node: ast.Set):
        return {self._is_hash(self.eval(elt)) for elt in node.elts}

    def eval_Dict(self, node: ast.Dict):
        return {self._is_hash(self.eval(k)): self.eval(v) for k, v in zip(node.keys, node.values)}


class NumPyEvalMixin(AbstractEval):
    """Mixin to allow safe evaluation of NumPy constants and functions via np.<name>."""

    def _is_public(self, name: str) -> bool:
        return not name.startswith("_")

    def _resolve_np_attr(self, node: ast.Attribute | ast.Name) -> Any:
        """
        Resolve an ast.Attribute chain rooted at Name('np') to an attribute on numpy.
        Disallows private names (underscore-prefixed) anywhere in the chain.
        """
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        else:
            raise ValueError("Only attributes rooted at 'np' are allowed")

        if parts[-1] != "np":
            if hasattr(np, parts[-1]):
                parts.append("np")
            else:
                raise ValueError("Only the 'np' namespace is allowed")

        parts.reverse()  # now ["np", "sin", ...]
        for p in parts[1:]:
            if not self._is_public(p):
                raise ValueError(f"Private NumPy attribute '{p}' not allowed")
        obj = np
        for p in parts[1:]:
            try:
                obj = getattr(obj, p)
            except AttributeError:
                raise ValueError(f"'np.{'.'.join(parts[1:])}' is not available")
        return obj

    def _resolve_np_name(self, node: ast.Name) -> Any:
        if node.id == "np":
            return np
        elif hasattr(np, node.id) and self._is_public(node.id):
            return getattr(np, node.id)
        else:
            raise ValueError(f"Name 'np.{node.id}' is not available")

    def eval_Name(self, node: ast.Name) -> Any:
        obj = self._resolve_np_name(node)
        if isinstance(obj, (int, float, np.integer, np.floating)):
            return obj
        raise ValueError("Only numeric NumPy constants can be used without calling")

    def eval_Attribute(self, node: ast.Attribute) -> Any:
        obj = self._resolve_np_attr(node)
        # Allow numeric constants (e.g., np.pi)
        if isinstance(obj, (int, float, np.integer, np.floating)):
            return obj
        raise ValueError("Only numeric NumPy constants can be used without calling")

    def eval_Call(self, node: ast.Call) -> Any:
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")
        if isinstance(node.func, ast.Attribute):
            func = self._resolve_np_attr(node.func)
        elif isinstance(node.func, ast.Name):
            func = self._resolve_np_name(node.func)
        else:
            raise ValueError("Only calls to 'np.<function>(...)' are allowed")
        if not callable(func):
            raise ValueError("Target is not callable")
        args = [self.eval(arg) for arg in node.args]
        # Ensure args are numeric scalars
        for a in args:
            if not isinstance(a, (int, float, np.integer, np.floating)):
                raise ValueError("Only numeric scalar arguments are allowed")
        return func(*args)


class FuncEvalMixin(AbstractEval):
    """Mixin to allow safe evaluation of a small set of built-in Python functions."""

    allowed_functions: dict[str, Callable[..., Any]] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    def eval_Name(self, node: ast.Name) -> Any:
        if node.id in self.allowed_functions:
            return self.allowed_functions[node.id]
        raise ValueError(f"Name '{node.id}' is not an allowed function")

    def eval_Call(self, node: ast.Call) -> Any:
        # if node.keywords:
        #     raise ValueError("Keyword arguments are not allowed")
        func = self.eval(node.func)
        if not callable(func):
            raise ValueError("Target is not callable")
        args = [self.eval(arg) for arg in node.args]
        kwargs = {str(kw.arg): self.eval(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)


# Composed evaluators
class MathEvaluator(ASTEvaluatorBase[int | float], MathEvalMixin):
    pass


class ContainerEvaluator(ASTEvaluatorBase[tuple | list | set | dict], ContainerEvalMixin):
    pass


class NumPyEvaluator(ASTEvaluatorBase, NumPyEvalMixin):
    pass


class FuncEvaluator(ASTEvaluatorBase, FuncEvalMixin, NumPyEvalMixin):
    def __init__(self, *args, allowed_functions=None, return_func=False, **kwargs):
        if allowed_functions is not None:
            self.allowed_functions = allowed_functions
        self.return_func = return_func
        super().__init__(*args, **kwargs)
        for key, value in kwargs.items():
            if callable(value):
                self.allowed_functions[key] = value
            if isinstance(value, dict):
                self.allowed_functions |= {k: v for k, v in value.items() if callable(v)}

    def eval_Name(self, node: ast.Name) -> Any:
        if not self.return_func:
            return NumPyEvalMixin.eval_Name(self, node)

        return self.allowed_functions.get(node.id, NumPyEvalMixin._resolve_np_name(self, node))

    def eval_Call(self, node: ast.Call) -> Any:
        # Try FuncEvalMixin first
        try:
            result = FuncEvalMixin.eval_Call(self, node)
        except ValueError:
            result = None
        if result is not None:
            return result

        # Then try NumPyEvalMixin
        try:
            result = NumPyEvalMixin.eval_Call(self, node)
        except ValueError:
            result = None
        if result is not None:
            return result

        # Nothing handled it
        raise ValueError("Unsupported function call")


class FullEvaluator(FuncEvaluator, ContainerEvalMixin, MathEvalMixin):
    def eval_Name(self, node: ast.Name) -> Any:
        try:
            return super().eval_Name(node)
        except ValueError:
            return node.id


def find_common_str(
    *strings: str,
    sep: str = "",
    junk: str | Callable[[str], bool] | None = None,
    by_block: bool = True,
    placeholder: str = "",
    retry: bool = True,
) -> tuple[str, list[str]]:
    """
    Returns the longest common substring from a list of strings.

    Parameters:
    strings (list of str): The list of strings to find the common substring from.

    Returns:
    str: The longest common substring found in all strings.
    """
    if not strings:
        return "", []
    elif len(strings) == 1:
        if not isinstance(strings[0], str) and len(strings[0]) > 1:
            strings = strings[0]
        else:
            return strings[0], list(strings)

    # Handle separator and junk character settings
    if not isinstance(sep, str):
        sep = ""
    if isinstance(junk, str):
        junk_str = junk

        def func(x):
            return x in junk_str

        junk = func

    # Initialize the common substring with the first string
    common_str = strings[0]
    sep_in_strings = True if sep and all(sep in s for s in strings) else False

    # Iterate through the remaining strings to find the common substring
    for string in strings[1:]:
        if by_block and sep_in_strings and junk is None:
            sequence = SequenceMatcher(junk, common_str.split(sep), string.split(sep))
            res = sequence.find_longest_match(
                0, len(common_str.split(sep)), 0, len(string.split(sep))
            )
            common_str = sep.join(common_str.split(sep)[res.a : res.a + res.size])
        else:
            sequence = SequenceMatcher(junk, common_str, string)
            if by_block:
                blocks = sequence.get_matching_blocks()
                res = max(blocks, key=lambda block: block.size)
            else:
                res = sequence.find_longest_match(0, len(common_str), 0, len(string))

            common_str = common_str[res.a : res.a + res.size]

    if not common_str and by_block and retry:
        return find_common_str(
            *strings, sep=sep, junk=junk, by_block=False, placeholder=placeholder
        )

    new_strings = [re.sub(common_str, placeholder, s).strip(sep) for s in strings]
    # Remove leading and trailing separators
    common_str = common_str.strip(sep)
    return common_str, new_strings


def common_substring(strings: list[str], sep: str = "") -> tuple[str, list[str]]:
    """
    Returns the longest common substring from a list of strings.

    Parameters:
    strings (list of str): The list of strings to find the common substring from.

    Returns:
    str: The longest common substring found in all strings.
    """
    if not strings:
        return "", strings
    split_strings = [[s for s in re.split(sep, st) if s != ""] for st in strings]
    # Find the shortest string in the list
    shortest_str = min(split_strings, key=len)

    # Check all substrings of the shortest string
    for length in range(len(shortest_str), 0, -1):
        for start in range(len(shortest_str) - length + 1):
            substring = shortest_str[start : start + length]
            if all(sep.join(substring) in sep.join(s) for s in split_strings):
                substring = sep.join(substring)
                residuals = [
                    (re.sub(substring + sep, "", sep.join(s)) if substring != sep.join(s) else "")
                    for s in split_strings
                ]
                return substring, residuals

    return sep.join(shortest_str), [
        (
            re.sub(sep.join(shortest_str) + sep, "", sep.join(s))
            if sep.join(shortest_str) != sep.join(s)
            else ""
        )
        for s in strings
    ]


# def str_in_list(key, keys, min_len=3):
#     """
#     Finds the closest matching string in a list of strings.

#     This function searches for the closest matching string in a list of strings (`keys`)
#     based on a given `key`. If an exact match is not found, it looks for partial matches
#     with a minimum length specified by `min_len`. The function returns the closest match
#     or the original key if no match is found.

#     Parameters:
#     key (str): The string to search for in the list.
#     keys (list of str): The list of strings to search within.
#     min_len (int, optional): The minimum length of the substring to consider for partial matches.
#                              Default is 3.

#     Returns:
#     str: The closest matching string from the list, or the original key if no match is found.
#     """
#     if key in keys:
#         # return if perfect match
#         return key
#     else:
#         if abs(min_len) >= len(key):
#             min_len = 0
#         n_keys = [k for k in keys if k in key and len(k) >= abs(min_len)] # check if a string in the list is wi
#         n = None
#         while len(n_keys) == 0 and abs(min_len) <= len(key[:n]):
#             n_keys = [k for k in keys if key[:n] in k]
#             if n is None:
#                 n = -1
#             else:
#                 n -= 1

#         if len(n_keys) == 0:
#             return key
#         elif isinstance(key, str):
#             n_keys.sort(key=lambda x: len(x) - len(key))
#         else:
#             n_keys.sort()
#         return n_keys[0]
if __name__ == "__main__":
    # %% Safe Eval test
    var_dict = "{A:1, B:2.0, C:True, D:a, E:'b', 0:v1, log(5):log(5), log:log, pi:pi}"
    var_list = "[1,2.0,True,a,'b',v1,log(5),log, pi]"
    var_list_open = "10,20.0,False,c,'d',v2,log10(5),log10, e"
    var_math = "1.2e3 + 5e2"
    MathEvaluator().parse("inf")

    res_1 = MathEvaluator().parse(var_math)
    res_2 = FullEvaluator().parse(var_math)
    res_3 = ContainerEvaluator().parse(var_list)
    res_4 = FullEvaluator().parse(var_list)
    res_5 = FullEvaluator(return_func=True).parse(var_list)
    res_6 = FullEvaluator().parse(var_list_open)
    res_7 = ContainerEvaluator().parse(var_dict)
    res_8 = FullEvaluator().parse(var_dict)
    # res_6 = FullEvaluator("raise").parse(var_list_open)
