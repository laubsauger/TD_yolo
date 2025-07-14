"""
TouchDesigner global type hints for Python linters
This file helps suppress false warnings about undefined TouchDesigner globals.
Import this at the top of TD scripts to provide type hints.
"""
from typing import Any

# TouchDesigner built-in globals
op: Any  # Access operators by path
me: Any  # Reference to current script operator
parent: Any  # Parent operator
ipar: Any  # Internal parameters
iop: Any  # Internal operators

# These are injected by TouchDesigner at runtime and won't exist outside TD