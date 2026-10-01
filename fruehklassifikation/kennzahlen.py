"""Kennzahlen der Frühklassifikation.

    Accuracy    Anteil der Reihen, deren entschiedene Klasse stimmt
    Earliness   mittlerer Anteil der Reihe, der bis zur Entscheidung gesehen wurde (kleiner = früher)
    HM          harmonisches Mittel aus Accuracy und 1 - Earliness (wie in TEASER), fasst beides in einer Zahl
    F1          Macro-F1, also der F1-Wert je Klasse gemittelt über alle Klassen
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

# Alle Kennzahlen werden auf so viele Nachkommastellen gerundet (wie in der Arbeit, auch für die Wahl von nu und m)
STELLEN = 4


def hm(accuracy: float, earliness: float) -> float:
    """Harmonisches Mittel aus Accuracy und 1 - Earliness."""
    fruehzeitigkeit = 1 - earliness
    return 2 * accuracy * fruehzeitigkeit / max(1e-12, accuracy + fruehzeitigkeit)


def bewerte(y_idx, klasse, earliness, n_klassen: int) -> dict:
    """Kennzahlen für wahre Klassen y_idx, entschiedene Klassen `klasse` (beide als Index) und Earliness je Reihe.

    Rückgabe: Accuracy, Earliness, HM, F1, Zahl der vorzeitig entschiedenen Reihen und Zahl aller Reihen.
    """
    y_idx = np.asarray(y_idx)
    klasse = np.asarray(klasse)
    earliness = np.asarray(earliness, dtype=float)

    accuracy = float(np.mean(y_idx == klasse))
    mittlere_earliness = float(np.mean(earliness))
    f1 = f1_score(y_idx, klasse, average="macro", labels=range(n_klassen), zero_division=0)

    return {"Accuracy": round(accuracy, STELLEN),
            "Earliness": round(mittlere_earliness, STELLEN),
            "HM": round(hm(accuracy, mittlere_earliness), STELLEN),
            "F1": round(float(f1), STELLEN),
            "vorzeitig": int(np.sum(earliness < 1.0)),
            "n": int(len(y_idx))}

