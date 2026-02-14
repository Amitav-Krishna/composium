"""Composium — Music toolkit for agents."""

from composium.analysis import analyze
from composium.instruments.piano import arrange_piano
from composium.instruments.guitar import arrange_guitar
from composium.instruments.drums import arrange_drums
from composium.instruments.edm import arrange_edm
from composium.compose import compose
from composium.render import render, layer, layer_with_envelope
from composium.notation import Note, Voice, Score, Analysis, Chord

__all__ = [
    "analyze",
    "arrange_piano",
    "arrange_guitar",
    "arrange_drums",
    "arrange_edm",
    "compose",
    "render",
    "layer",
    "layer_with_envelope",
    "Note",
    "Voice",
    "Score",
    "Analysis",
    "Chord",
]
