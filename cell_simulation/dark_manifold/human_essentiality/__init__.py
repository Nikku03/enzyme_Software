"""
Human Gene Essentiality Predictor

Adapted from the JCVI-syn3A "hedge fund" approach for human cancer cell lines
using DepMap CRISPR screening data.
"""

from .human_essentiality_predictor import HumanEssentialityPredictor

__all__ = ['HumanEssentialityPredictor']
