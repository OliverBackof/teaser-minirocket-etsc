"""UEA-Datensatz herunterladen und aufteilen.

Die Datensätze kommen aus dem UEA-Archiv (timeseriesclassification.com) und werden über aeon geladen. Beim ersten
Aufruf lädt aeon den Datensatz herunter, danach liegt er im Download-Ordner.

Aufteilung wie in der Arbeit:

    Archiv-TRAIN  ->  Train
    Archiv-TEST   ->  je Klasse zur Hälfte Validierung und Test (fester Startwert)

Die Reihen kommen als Liste von Arrays (Messpunkte x Kanäle, float32) zurück. Fehlende Werte werden 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Startwert für die Aufteilung des Archiv-TEST
SEED = 42


def als_liste(X) -> list[np.ndarray]:
    """aeon-Format (Reihen x Kanäle x Messpunkte) -> Liste von Arrays (Messpunkte x Kanäle), float32, NaN = 0."""
    return [np.nan_to_num(np.asarray(reihe, dtype=np.float32).T, nan=0.0) for reihe in X]


def teile_archiv_test(y_test: list) -> tuple[np.ndarray, np.ndarray]:
    """Teilt die Indizes des Archiv-TEST je Klasse zur Hälfte in Validierung und Test.

    Rückgabe: (Indizes Validierung, Indizes Test), jeweils aufsteigend sortiert.
    """
    alle = np.arange(len(y_test))
    i_val, i_test = train_test_split(alle, test_size=0.5, stratify=y_test, random_state=SEED)
    return np.sort(i_val), np.sort(i_test)


def lade_uea(name: str, ordner: Path) -> dict:
    """Lädt den UEA-Datensatz `name` (beim ersten Mal Download nach `ordner`) und teilt ihn auf.

    Rückgabe:
        {"train": (Reihen, Labels), "val": (Reihen, Labels), "test": (Reihen, Labels), "info": {...}}
    Vorgesehen sind multivariate Datensätze mit gleich langen Reihen.
    """
    from aeon.datasets import load_classification

    ordner = Path(ordner)
    ordner.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_classification(name, split="train", extract_path=str(ordner))
    X_test, y_test = load_classification(name, split="test", extract_path=str(ordner))

    X_train, X_test = als_liste(X_train), als_liste(X_test)
    y_train, y_test = [str(k) for k in y_train], [str(k) for k in y_test]

    formen = {reihe.shape for reihe in X_train + X_test}
    if len(formen) != 1:
        raise ValueError(f"{name}: Reihen unterschiedlicher Länge, nur gleich lange Datensätze vorgesehen")
    laenge, n_kanaele = formen.pop()

    i_val, i_test = teile_archiv_test(y_test)
    info = {"datensatz": name, "n_train": len(X_train), "n_val": len(i_val), "n_test": len(i_test),
            "laenge": int(laenge), "kanaele": int(n_kanaele), "klassen": sorted(set(y_train + y_test))}
    return {"train": (X_train, y_train),
            "val": ([X_test[i] for i in i_val], [y_test[i] for i in i_val]),
            "test": ([X_test[i] for i in i_test], [y_test[i] for i in i_test]),
            "info": info}
