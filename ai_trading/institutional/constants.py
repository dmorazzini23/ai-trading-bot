"""Institutional-specific trading constants.

The institutional risk specification mandates that the Kelly fraction cap
is at least 25% to ensure sufficient capital deployment. This module
records that requirement explicitly.
"""

# Minimum Kelly fraction cap required by institutional risk standards.
MAX_KELLY_FRACTION: float = 0.25
