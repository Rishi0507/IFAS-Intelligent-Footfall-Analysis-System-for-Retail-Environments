"""Inference pipeline: person detection + gender/age classification + analytics."""
from .tracker import Track, IoUTracker
from .analytics import build_analytics

__all__ = ["Track", "IoUTracker", "build_analytics"]
