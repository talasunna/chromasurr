from __future__ import annotations
import os
import sys
from datetime import datetime
from typing import List

# --- Path setup: add src/ so autodoc can import chromasurr.* ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

# --- Project info ---
project = "chromasurr"
author = "Tala Al-Sunna"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

# --- Extensions ---
extensions: List[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # NumPy-style docstrings
    "sphinx_autodoc_typehints",     # render type hints
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

# Autodoc / autosummary defaults
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "special-members": "__call__",
}

# Napoleon for NumPy-style
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_rtype = False

# Type hints rendering
typehints_fully_qualified = False
typehints_use_rtype = False

# Intersphinx: second tuple element must be a string (inventory) or None
# Using None makes Sphinx look for 'objects.inv' at the URL.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# --- HTML ---
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]

# Optional: catch bad refs
# nitpicky = True
