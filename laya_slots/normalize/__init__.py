"""Normalisation des valeurs de slots (dates relatives, nombres, heures).

Toute valeur gold ou candidate passe par `normalize_value` avant comparaison (CONTRAT.md, §3).
"""
from .values import normalize_value

__all__ = ["normalize_value"]
