"""Shim: the original app resolves asset paths from its install root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

__all__ = ['ROOT']
