# -*- coding: utf-8 -*-
"""
Init File.

@author: j2cle
Created on Thu Sep 19 11:17:33 2024
"""

from .string_evaluation import (
    eval_string,
    common_substring,
    str_in_list,
)
from .string_manipulation import (
    sci_note,
    re_not,
    slugify,
    eng_not,
)

__all__ = [
    "eval_string",
    "common_substring",
    "str_in_list",
    "sci_note",
    "re_not",
    "slugify",
    "eng_not",
]
