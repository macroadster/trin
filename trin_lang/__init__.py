"""Trin language package — ternary control (gate | pure | each)."""

from .interpreter import Interpreter, run_file, run_source
from .typecheck import typecheck_program, typecheck_source

__all__ = [
    "Interpreter",
    "run_file",
    "run_source",
    "typecheck_program",
    "typecheck_source",
]
__version__ = "0.3.0"
