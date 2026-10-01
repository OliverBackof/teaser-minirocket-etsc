"""Abbruchregel: wann über eine Reihe entschieden wird.

Symmetrische Regel wie in TEASER: Sobald an m Schnitten in Folge dieselbe Klasse vorhergesagt und vom Master als
verlässlich eingestuft wurde, ist die Reihe entschieden. Eine unverlässliche Ausgabe oder ein Wechsel der Klasse
beginnt die Zählung von vorn.

Kommt es bis zum letzten Schnitt zu keiner Entscheidung, gilt die Vorhersage des letzten Schnitts.

Earliness = Präfixlänge bei der Entscheidung / Reihenlänge (1, wenn erst am Ende entschieden wird).
"""
from __future__ import annotations

import numpy as np


def entscheide_reihe(schritte: list, laenge: int, m: int) -> tuple[int, float]:
    """Entscheidung für eine Reihe.

    schritte: Liste (Präfixlänge, vorhergesagte Klasse als Index, verlässlich) über alle Schnitte, zeitlich
    geordnet. Rückgabe: (entschiedene Klasse als Index, Earliness).
    """
    laenge = max(1, int(laenge))
    in_folge = 0                 # Zahl der verlässlichen gleichen Vorhersagen direkt hintereinander
    letzte_klasse = None         # Klasse der letzten verlässlichen Vorhersage

    for praefix_laenge, klasse, ist_verlaesslich in schritte:
        if not ist_verlaesslich:
            in_folge, letzte_klasse = 0, None
        elif klasse == letzte_klasse:
            in_folge += 1
        else:
            in_folge, letzte_klasse = 1, klasse

        if in_folge >= m:
            return int(klasse), min(1.0, praefix_laenge / laenge)

    # Keine vorzeitige Entscheidung: Vorhersage des letzten Schnitts, die ganze Reihe wurde gesehen
    _, letzte_vorhersage, _ = schritte[-1]
    return int(letzte_vorhersage), 1.0


def entscheide_alle(schritte_je_reihe: list, laenge: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Entscheidung für alle Reihen gleicher Länge. Rückgabe: (Klassen als Index, Earliness) als Arrays."""
    entscheidungen = [entscheide_reihe(schritte, laenge, m) for schritte in schritte_je_reihe]
    klassen = np.array([klasse for klasse, _ in entscheidungen])
    earliness = np.array([frueh for _, frueh in entscheidungen], dtype=float)
    return klassen, earliness
