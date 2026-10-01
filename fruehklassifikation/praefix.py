"""Schnitte und Präfixe.

Eine Reihe ist ein Array der Form (Messpunkte x Kanäle). Das Modell sieht eine Reihe nicht auf einmal, sondern
an mehreren Schnitten nacheinander. Am Schnitt j kennt es das Präfix, also die ersten s_j = j * w Messpunkte:

    Reihe:     x1 x2 x3 ... ------------------------------------------------------>  Zeit
    Schnitte:          s_1 = w        s_2 = 2w        s_3 = 3w       ...

Bevor ein Präfix an den Slave geht, wird es je Kanal z-normiert (Mittelwert 0, Standardabweichung 1).
"""
from __future__ import annotations

import numpy as np

# Einstellungen

# Anzahl der Schnitte je Reihe, daraus folgt der Schnittabstand w = Reihenlänge / 20 (wie in TEASER)
SCHNITTE_JE_REIHE = 20

# Kürzere Präfixe kann MiniROCKET nicht falten (Kernlänge 9). Diese Schnitte entfallen, außer die ganze Reihe ist
# kürzer (z. B. PenDigits mit 8 Punkten). Dann bleiben alle Schnitte, und die Präfixe werden links mit dem ersten
# Messwert auf diese Länge aufgefüllt.
MINDESTLAENGE_PRAEFIX = 9

# Kleinste Standardabweichung, durch die geteilt wird. Darunter gilt ein Kanal als konstant.
EPS = 1e-8


# Schnitte

def schnittabstand(laenge: int) -> int:
    """Schnittabstand w = Reihenlänge / 20, mindestens 1."""
    return max(1, int(laenge) // SCHNITTE_JE_REIHE)


def schnitte(w: int, laenge: int) -> list[tuple[int, int]]:
    """Alle Schnitte einer Reihe als Liste (j, s_j) mit s_j = j * w.

    Die Schnitte laufen bis zur Reihenlänge. Präfixe mit weniger als MINDESTLAENGE_PRAEFIX Punkten entfallen,
    außer die Reihe selbst ist kürzer. Dann bleiben alle Schnitte (siehe auffuellen).
    """
    alle = [(j, j * w) for j in range(1, laenge // w + 1)]
    if laenge < MINDESTLAENGE_PRAEFIX:
        return alle
    return [(j, praefix_laenge) for j, praefix_laenge in alle if praefix_laenge >= MINDESTLAENGE_PRAEFIX]


def auffuellen(praefix: np.ndarray) -> np.ndarray:
    """Füllt ein Präfix (Messpunkte x Kanäle) links mit seinem ersten Messwert auf MINDESTLAENGE_PRAEFIX Punkte auf.

    Längere Präfixe bleiben unverändert.
    """
    fehlend = MINDESTLAENGE_PRAEFIX - len(praefix)
    if fehlend <= 0:
        return praefix
    return np.concatenate([np.repeat(praefix[:1], fehlend, axis=0), praefix], axis=0)


# Normierung

def normiere_praefix(praefix: np.ndarray) -> np.ndarray:
    """z-Normierung eines Präfixes je Kanal.

    Eingabe: Präfix (Messpunkte x Kanäle). Rückgabe: normiertes Präfix (Kanäle x Messpunkte), float32, so wie
    MiniROCKET es erwartet. Mittelwert und Standardabweichung werden in float32 über das Präfix berechnet.
    Konstante Kanäle werden 0, ebenso NaN und unendliche Werte.
    """
    mittelwert = np.asarray(praefix.mean(axis=0, dtype=np.float32), dtype=np.float32)
    streuung = np.asarray(praefix.std(axis=0, dtype=np.float32), dtype=np.float32)

    werte = np.asarray(praefix, dtype=np.float32)
    normiert = (werte - mittelwert[None, :]) / np.maximum(streuung, EPS)[None, :]
    normiert = normiert.astype(np.float32, copy=False)

    normiert[:, streuung < EPS] = 0.0
    np.nan_to_num(normiert, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return normiert.T


def stapel_praefixe(X: list[np.ndarray], praefix_laenge: int) -> np.ndarray:
    """Normierte Präfixe aller Reihen als ein Array (Reihen x Kanäle x Messpunkte), float32.

    Präfixe unter MINDESTLAENGE_PRAEFIX Punkten werden vor der Normierung links aufgefüllt (siehe auffuellen).
    """
    normiert = [normiere_praefix(auffuellen(np.asarray(reihe, dtype=np.float32)[:praefix_laenge])) for reihe in X]
    return np.stack(normiert, axis=0).astype(np.float32, copy=False)
