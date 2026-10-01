"""Frühklassifikation auf UEA-Datensätzen: Hauptskript.

Für jeden gewählten Datensatz:

    1. Datensatz laden und aufteilen (daten.py)
    2. Modell trainieren, nu und m auf der Validierung nach HM wählen (fruehklassifikation)
    3. einmal auf dem Test auswerten
    4. Ergebnis als ergebnisse/<Datensatz>.json schreiben

Am Ende entsteht ergebnisse/uebersicht.md mit einer Tabelle aller Datensätze.

Kennzahlen auf dem Test: Accuracy am Laufende (S = 1) sowie Accuracy, Earliness, HM und Macro-F1 beim Frühabbruch.

Aufruf:
    python experiment.py                                          alle Standard-Datensätze
    python experiment.py --datensaetze BasicMotions               ein Datensatz
    python experiment.py --datensaetze Epilepsy NATOPS --geraet gpu --speichern
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from daten import lade_uea
from fruehklassifikation import Fruehklassifikator, bewerte
from fruehklassifikation.modell import paket_versionen

# Einstellungen

ORDNER = Path(__file__).resolve().parent

# Standard-Datensätze (sieben Beispiele aus den 26 Datensätzen der Arbeit), per --datensaetze änderbar
DATENSAETZE = ["ArticularyWordRecognition", "BasicMotions", "Epilepsy", "Handwriting", "Heartbeat", "NATOPS",
               "SelfRegulationSCP1"]

DATEN = ORDNER / "uea_daten"          # Download-Ordner der Datensätze
ERGEBNISSE = ORDNER / "ergebnisse"    # Ordner für die Ergebnisse

# Einstellungen des Slaves
N_KERNELS = 5000                      # MiniROCKET-Merkmale
C = 1.0                               # Regularisierung der logistischen Regression
MAX_ITER = 150                        # Iterationen der logistischen Regression


# Einen Datensatz rechnen

def rechne(name: str, daten_ordner: Path, ziel: Path, geraet: str, speichern: bool) -> dict:
    """Trainiert und bewertet das Modell auf einem Datensatz und schreibt ergebnisse/<name>.json."""
    t_start = time.time()

    # 1. Daten
    daten = lade_uea(name, daten_ordner)
    info = daten["info"]
    print(f"   {name}: Train {info['n_train']}, Val {info['n_val']}, Test {info['n_test']} Reihen, "
          f"Länge {info['laenge']}, {info['kanaele']} Kanäle, {len(info['klassen'])} Klassen", flush=True)

    # 2. Training mit Wahl von nu und m auf der Validierung
    modell = Fruehklassifikator(n_kernels=N_KERNELS, C=C, max_iter=MAX_ITER, geraet=geraet)
    X_train, y_train = daten["train"]
    X_val, y_val = daten["val"]
    modell.fit(X_train, y_train, X_val, y_val)

    # 3. Auswertung auf dem Test, einmal mit Frühabbruch und einmal am Laufende (S = 1)
    X_test, y_test = daten["test"]
    vorhersage = modell.vorhersage(X_test)
    index_der_klasse = {klasse: i for i, klasse in enumerate(modell.klassen)}
    y_idx = [index_der_klasse[k] for k in y_test]
    n_klassen = len(modell.klassen)

    fruehabbruch = bewerte(y_idx, [index_der_klasse[k] for k in vorhersage["klasse"]],
                           vorhersage["earliness"], n_klassen)
    laufende = bewerte(y_idx, [index_der_klasse[k] for k in vorhersage["klasse_s1"]],
                       np.ones(len(y_test)), n_klassen)

    # 4. Ergebnisdatei
    info.update(w=modell.w_, schnitte=[sl["praefix_laenge"] for _, sl in sorted(modell.slaves.items())])
    ergebnis = {
        "datensatz": name,
        "S1_accuracy": laufende["Accuracy"],
        "S1_f1": laufende["F1"],
        "fruehabbruch": fruehabbruch,
        "auswahl": modell.auswahl,
        "info": info,
        "laufzeit_s": round(time.time() - t_start, 1),
        "geraet": geraet,
        "einstellung": {"n_kernels": N_KERNELS, "C": C, "max_iter": MAX_ITER},
        "versionen": paket_versionen(),
        "test_detail": {"y": y_test, "klasse": vorhersage["klasse"], "klasse_s1": vorhersage["klasse_s1"],
                        "earliness": [round(float(e), 4) for e in vorhersage["earliness"]]},
    }
    ziel.mkdir(parents=True, exist_ok=True)
    pfad = ziel / f"{name}.json"
    pfad.write_text(json.dumps(ergebnis, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"   geschrieben: {pfad} (existiert: {pfad.exists()})", flush=True)

    if speichern:
        modell_pfad = modell.speichern(ziel / "modelle" / name)
        print(f"   Modell gespeichert: {modell_pfad} (existiert: {modell_pfad.exists()})", flush=True)
    return ergebnis

def tabelle(ergebnisse: list[dict]) -> list[str]:
    """Markdown-Tabelle mit einer Zeile je Datensatz."""
    zeilen = ["| Datensatz | S=1 Acc | Frühabbruch Acc | Earliness | HM | F1 | nu | m | Laufzeit [s] |",
              "|---|---|---|---|---|---|---|---|---|"]
    for e in ergebnisse:
        f = e["fruehabbruch"]
        zeilen.append(f"| {e['datensatz']} | {e['S1_accuracy']:.4f} "
                      f"| {f['Accuracy']:.4f} | {f['Earliness']:.4f} | {f['HM']:.4f} | {f['F1']:.4f} "
                      f"| {e['auswahl']['nu']} | {e['auswahl']['m']} | {e['laufzeit_s']:.0f} |")
    return zeilen


def schreibe_uebersicht(ergebnisse: list[dict], ziel: Path) -> Path:
    """Schreibt ergebnisse/uebersicht.md mit der Ergebnistabelle."""
    md = ["# Ergebnisse auf den UEA-Datensätzen (Testdaten)", ""]
    md += tabelle(ergebnisse)
    pfad = ziel / "uebersicht.md"
    pfad.write_text("\n".join(md) + "\n", encoding="utf-8")
    return pfad

def main() -> None:
    """Liest die Argumente und rechnet alle gewählten Datensätze nacheinander."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datensaetze", nargs="+", default=DATENSAETZE, help="Namen der UEA-Datensätze")
    parser.add_argument("--geraet", default="cpu", choices=["cpu", "gpu"], help="Gerät für MiniROCKET")
    parser.add_argument("--daten", default=str(DATEN), help="Download-Ordner der Datensätze")
    parser.add_argument("--ergebnisse", default=str(ERGEBNISSE), help="Ordner für die Ergebnisse")
    parser.add_argument("--speichern", action="store_true", help="Modelle unter <ergebnisse>/modelle/ speichern")
    args = parser.parse_args()

    ziel = Path(args.ergebnisse)
    ergebnisse = []
    n = len(args.datensaetze)
    t_start = time.time()

    for i, name in enumerate(args.datensaetze, 1):
        vergangen = time.time() - t_start
        eta = f"ETA {vergangen / (i - 1) * (n - i + 1) / 60:.1f} min" if i > 1 else "ETA offen"
        print(f"[{i}/{n}] {100 * (i - 1) / n:.0f} % {eta} === {name}", flush=True)
        ergebnisse.append(rechne(name, Path(args.daten), ziel, args.geraet, args.speichern))

    print(f"[{n}/{n}] 100 % fertig nach {(time.time() - t_start) / 60:.1f} min\n", flush=True)
    print("\n".join(tabelle(ergebnisse)), flush=True)
    pfad = schreibe_uebersicht(ergebnisse, ziel)
    print(f"\ngeschrieben: {pfad} (existiert: {pfad.exists()})", flush=True)


if __name__ == "__main__":
    main()
