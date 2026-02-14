"""
Subagent registry — exports all subagent classes.
"""

from .drums import DrumsSubagent
from .melody import MelodySubagent
from .vocal import VocalSubagent

SUBAGENT_REGISTRY = {
    "drums": DrumsSubagent,
    "melody": MelodySubagent,
    "vocal": VocalSubagent,
}

__all__ = ["DrumsSubagent", "MelodySubagent", "VocalSubagent", "SUBAGENT_REGISTRY"]
