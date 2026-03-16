"""Compatibility shim for Module B Part 1 database imports.

Usage:
    from drug_metabolism_db import DRUG_DATABASE, get_drug, list_drugs
"""

from enzyme_software.calibration.drug_metabolism_db import *  # noqa: F401,F403
