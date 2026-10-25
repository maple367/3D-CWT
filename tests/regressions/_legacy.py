"""Run allowlisted, checked-in legacy method ASTs without importing legacy.

This is not a sandbox for untrusted code. Only the three fixed local classes
below are supported. Method bodies/locations are preserved, not reimplemented;
imports, module statements, constructors, decorators and base classes are not
executed. Keeping each original class name preserves private-name mangling.
"""

import ast
from pathlib import Path

import numpy as np


# An explicit opt-in to historical defects, not a claim about the current HEAD.
LEGACY_BASELINE = "520f4d8-unfixed"
_ROOT = Path(__file__).resolve().parents[2]
_TARGETS = {
    "CWT_solver": ("model/_Model.py", ("_chi_func",)),
    "Model": ("model/_Model.py", ("beta_z_func_higher_order", "Green_func_higher_order")),
    "xi_calculator_DFT": ("calculator/_calc.py", ("_cal",)),
}


def legacy_class(name):
    """Compile original selected methods in an empty class of the same name.

    Callers supply only deterministic dependencies via instance attributes.
    Unknown targets and source structure drift fail closed instead of silently
    skipping a regression. Only the selected original method ASTs are compiled.
    """
    relative_path, method_names = _TARGETS[name]
    path = _ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(classes) != 1:
        raise AssertionError(f"Expected exactly one legacy class {name} in {path}")
    methods = [node for node in classes[0].body
               if isinstance(node, ast.FunctionDef) and node.name in method_names]
    if sorted(node.name for node in methods) != sorted(method_names):
        raise AssertionError(f"Legacy methods changed: {name}.{method_names}")
    if any(method.decorator_list for method in methods):
        raise AssertionError("Legacy extraction does not execute decorators")
    isolated = ast.ClassDef(name=name, bases=[], keywords=[], body=methods,
                           decorator_list=[], type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
    namespace = {"np": np, "__name__": "isolated_legacy_regression"}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]
