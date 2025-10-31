---
applyTo: "**/*.py,**/*.pyi"
---
# Project coding standards for Python and python stubs

Apply the [general coding guidelines](./general-coding.instructions.md) to all code.

## Python Guidelines
- Avoid  using single letter variable names except within comprehensions or lambda functions.
- Use `snake_case` for function and variable names.
- Use `PascalCase` for class names.
- Use `ALL_CAPS` for constants.
- Use type hints for function parameters and return types consistent with Python 3.10+.
- Use `__init__.py` files to mark directories as Python packages.
- Avoid single use variables.
- Use comments to explain complex or non-obvious code sections.
- Prioritize performance and direct solutions over extra variables or Pythonic loops for clarity.
- Minimize class and function name length, maximizing conciseness while maintaining clarity.
- Prefer minimal, direct code over stylistic or abstract solutions unless there is a clear benefit for research.
- Only recommend changes that improve performance, maintainability, or scientific clarity.
- Avoid suggesting refactoring or abstraction unless it is necessary for the research workflow.

## Data Handling
- Never use pandas.DataFrame.iterrows() unless explicitly told (or code already uses it).
- When working with data contained in matrix style objects from libraries like `numpy`, `pandas`, or `xarray`, maximize the use of built-in methods, functions, and indexing. Avoid using loops or comprehensions unless absolutely necessary, as these libraries are optimized for performance with their built-in operations.

## Error Handling
- Use `try/except` blocks for error handling.
- Never use bare `except:` clauses; specify the exception type.
- Never use `Exception`; specify the exception type.
- Always log errors with contextual information.
- Use `logging` module for logging errors and information if imported.
- Avoid using `print` statements for error handling or debugging.

## Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.
- Use one-line docstrings for class properties and nested functions.
- Use [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) style docstrings for modules, classes, functions and methods.
- Limit docstring line length to 100 characters.
- Wrap long lists with `# fmt: off` and `# fmt: on` comments to prevent automatic formatting.

## Testing
- Use `pytest` for testing.
- Use `**kwargs` type parameters (suitably named) for passing multiple arguments from `pytest.mark.parametrize`, unless not supported or only one argument is needed.
- When defining `**kwargs`, skip parameters that are unrelated to the specific parameter test case or unchanged from the default.
- Never use `kwargs` as a parameter name.
- Never generate data in a `pytest.mark.parametrize`.
- When using `pytest.mark.parametrize`, always include a 'description' for each test case.
- If `check_result_type_and_print` is imported, always include it immediately after the primary function call in the test.
- If `check_result_type_and_print` is not imported, always include a `print` statement after the primary function but before any `assert`.
- Where applicable, use `result` as the variable name for the result of the primary function call in tests.
- When generating new tests, include `clarifying comments` for each test case (especially when using `pytest.mark.parametrize`) that clarifies what specific logic, code path, or condition is being tested (e.g., "tests the `if x < 0` branch in `my_function`").
- For non-parameterized tests, add `clarifying comments` before each `assert` statement explaining what aspect of the function or which code branch is being verified.
- These `clarifying comments` are intended for clarity during review and may be removed later.

## Python Stub Guidelines
- Use `*.pyi` files for type stubs.