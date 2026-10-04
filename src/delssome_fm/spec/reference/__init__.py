"""Hand-written reference cards: the five of generation.md §10 (step 0) plus MPR and
Jansen-Rit, added on 2026-10-05.

These are the correctness gate for the spec layer: if the template cannot express them
exactly, the template is wrong. Each YAML file documents how its card parameters map to the
published parameters (some are rescaled or shifted by the canonicalisation of generation.md
§3).
"""

from pathlib import Path

from delssome_fm.spec.card import ModelCard, load_card

REFERENCE_DIR = Path(__file__).resolve().parent
REFERENCE_NAMES: tuple[str, ...] = ("linear", "mfm", "fic", "wilson_cowan", "hopf", "mpr",
                                    "jansen_rit")


def load_reference(name: str) -> ModelCard:
    if name not in REFERENCE_NAMES:
        raise KeyError(f"no reference card {name!r}; available: {REFERENCE_NAMES}")
    return load_card(REFERENCE_DIR / f"{name}.yaml")
