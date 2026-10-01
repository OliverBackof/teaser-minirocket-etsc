"""Fruehklassifikator: das komplette Modell aus Slaves, Mastern und Abbruchregel.

Ablauf beim Training (fit):

    1. Schnitte festlegen (Schnittabstand w = Reihenlänge / 20)
    2. je Schnitt einen Slave auf den Trainingsreihen trainieren (slave.py)
    3. für jedes nu aus NU_GITTER je Schnitt einen Master trainieren (master.py)
    4. für jedes Paar (nu, m) die Abbruchregel auf den Validierungsreihen anwenden (abbruch.py)
    5. das Paar mit dem höchsten HM behalten

Ablauf bei der Vorhersage (vorhersage):

    je Schnitt: Slave -> Wahrscheinlichkeiten q -> Master -> verlässlich ja/nein
    danach:     Abbruchregel mit dem gewählten m -> Klasse und Earliness je Reihe

Ein gespeichertes Modell ist ein Ordner mit drei Dateien:

    slaves.joblib      {j: Slave-dict} je Schnitt j, siehe slave.py
    masters.joblib     {j: Master-dict} je Schnitt j, siehe master.py
    einstellung.json   Einstellungen, Klassen, gewähltes nu und m, Schnitte, Paketversionen
"""
from __future__ import annotations

import json
import platform
import time
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np

from . import abbruch, kennzahlen, master, praefix, slave

# Einstellungen

# Kandidaten, aus denen nu (Master) und m (Abbruchregel) auf den Validierungsdaten gewählt werden
NU_GITTER = [0.0001, 0.001, 0.003, 0.01, 0.05]
M_WERTE = [1, 2, 3, 4, 5]

# Pakete, deren Versionen mit dem Modell gespeichert werden
PAKETE = ("numpy", "scikit-learn", "torch", "tsai", "joblib")

# Kompressionsstufe beim Speichern der joblib-Dateien (0 = keine, 9 = stärkste)
KOMPRESSION = 3


# Hilfsfunktionen

def fortschritt(text: str, erledigt: int, gesamt: int, t_start: float) -> str:
    """Fortschrittszeile mit Prozent und geschätzter Restzeit."""
    restzeit = (time.time() - t_start) / max(1, erledigt) * (gesamt - erledigt)
    prozent = 100 * erledigt / max(1, gesamt)
    return f"   {text} [{erledigt}/{gesamt}] {prozent:.1f} % ETA {restzeit:.0f} s"


def paket_versionen() -> dict:
    """Versionen von Python und den verwendeten Paketen."""
    versionen = {"python": platform.python_version()}
    for paket in PAKETE:
        try:
            versionen[paket] = metadata.version(paket)
        except metadata.PackageNotFoundError:
            versionen[paket] = None
    return versionen


# Modell

class Fruehklassifikator:
    """Frühklassifikation multivariater Zeitreihen gleicher Länge nach dem TEASER-Prinzip.

    Parameter:
        w           Schnittabstand in Messpunkten, None = Reihenlänge / 20
        n_kernels   Zahl der MiniROCKET-Merkmale je Slave (wird auf ein Vielfaches von 84 abgerundet)
        C           Regularisierung der logistischen Regression (kleiner = stärker)
        max_iter    Iterationen der logistischen Regression
        geraet      "cpu" oder "gpu" für MiniROCKET
        ausgabe     Fortschritt auf der Konsole ausgeben

    Nach fit stehen zur Verfügung: klassen, w_ (verwendeter Schnittabstand), nu_ und m_ (gewählt auf den
    Validierungsdaten) und auswahl (Kennzahlen aller geprüften Paare aus nu und m).
    """

    def __init__(self, w: int | None = None, n_kernels: int = 5000, C: float = 1.0, max_iter: int = 150,
                 geraet: str = "cpu", ausgabe: bool = True):
        self.w = w
        self.n_kernels = n_kernels
        self.C = C
        self.max_iter = max_iter
        self.geraet = geraet
        self.ausgabe = ausgabe

        # werden beim Training bzw. Laden gesetzt
        self.slaves: dict = {}
        self.masters: dict = {}
        self.klassen: list = []
        self.laenge = None
        self.n_kanaele = None
        self.w_ = None
        self.nu_ = None
        self.m_ = None
        self.auswahl: dict = {}

    # Hilfsmethoden

    def _log(self, text: str) -> None:
        if self.ausgabe:
            print(text, flush=True)

    def _klassenindizes(self, y) -> np.ndarray:
        """Labels -> Indizes in self.klassen."""
        index_der_klasse = {klasse: i for i, klasse in enumerate(self.klassen)}
        return np.array([index_der_klasse[klasse] for klasse in y])

    def _schritte(self, q_je_schnitt: dict, masters: dict) -> list:
        """Verlauf je Reihe über alle Schnitte: Liste (Präfixlänge, vorhergesagte Klasse, verlässlich).

        q_je_schnitt: {j: Wahrscheinlichkeiten q (Reihen x Klassen)}. Das ist die Eingabe der Abbruchregel.
        """
        n_reihen = len(next(iter(q_je_schnitt.values())))
        schritte = [[] for _ in range(n_reihen)]
        for j, q in sorted(q_je_schnitt.items()):
            ist_verlaesslich = master.verlaesslich(masters[j], q)
            praefix_laenge = self.slaves[j]["praefix_laenge"]
            for i in range(n_reihen):
                schritte[i].append((praefix_laenge, int(np.argmax(q[i])), bool(ist_verlaesslich[i])))
        return schritte

    # Training

    def _trainiere_slaves(self, X, y_idx, X_val, geraet) -> tuple[dict, dict]:
        """Trainiert je Schnitt einen Slave.

        Rückgabe: Wahrscheinlichkeiten q je Schnitt auf den Trainingsreihen (für die Master) und auf den
        Validierungsreihen (für die Wahl von nu und m).
        """
        alle_schnitte = praefix.schnitte(self.w_, self.laenge)
        labels = [self.klassen[k] for k in y_idx]
        q_train, q_val = {}, {}
        t_start = time.time()

        for n, (j, praefix_laenge) in enumerate(alle_schnitte, 1):
            X_praefix = praefix.stapel_praefixe(X, praefix_laenge)
            self.slaves[j], q_train[j] = slave.trainiere_slave(
                X_praefix, labels, self.klassen, praefix_laenge, self.n_kernels, self.C, self.max_iter, geraet)

            X_val_praefix = praefix.stapel_praefixe(X_val, praefix_laenge)
            q_val[j] = slave.slave_wahrscheinlichkeiten(self.slaves[j], X_val_praefix, geraet)
            self._log(fortschritt(f"Slave j={j} (s={praefix_laenge})", n, len(alle_schnitte), t_start))
        return q_train, q_val

    def fit(self, X, y, X_val, y_val) -> "Fruehklassifikator":
        """Trainiert Slaves und Master und wählt nu und m mit dem höchsten HM auf den Validierungsdaten.

        X, X_val: Listen von Arrays (Messpunkte x Kanäle), alle gleich lang. y, y_val: Labels.
        Bei Gleichstand im HM gewinnt das erste Paar (nu aufsteigend, innerhalb eines nu m aufsteigend).
        """
        self.laenge, self.n_kanaele = np.shape(X[0])
        self.klassen = sorted(set(y))
        y_idx, y_val_idx = self._klassenindizes(y), self._klassenindizes(y_val)
        self.w_ = int(self.w) if self.w else praefix.schnittabstand(self.laenge)
        geraet = slave.torch_geraet(self.geraet)

        self._log(f"   Training: {len(X)} Reihen, {self.n_kanaele} Kanäle, Länge {self.laenge}, "
                  f"{len(self.klassen)} Klassen, w={self.w_}, Validierung {len(X_val)} Reihen")

        # Schritt 1 und 2: Slaves
        q_train, q_val = self._trainiere_slaves(X, y_idx, X_val, geraet)

        # Schritt 3 und 4: für jedes nu die Master trainieren, für jedes m die Validierung bewerten
        kennzahlen_je_paar = {}
        masters_je_nu = {}
        t_start = time.time()
        for n, nu in enumerate(NU_GITTER, 1):
            masters_je_nu[nu] = {j: master.trainiere_master(q, y_idx, nu) for j, q in q_train.items()}
            schritte = self._schritte(q_val, masters_je_nu[nu])
            for m in M_WERTE:
                klasse, earliness = abbruch.entscheide_alle(schritte, self.laenge, m)
                kennzahlen_je_paar[(nu, m)] = kennzahlen.bewerte(y_val_idx, klasse, earliness, len(self.klassen))
            self._log(fortschritt(f"Master nu={nu}", n, len(NU_GITTER), t_start))

        # Schritt 5: bestes Paar (max liefert bei Gleichstand das erste)
        self.nu_, self.m_ = max(kennzahlen_je_paar, key=lambda paar: kennzahlen_je_paar[paar]["HM"])
        self.masters = masters_je_nu[self.nu_]
        bestes = kennzahlen_je_paar[(self.nu_, self.m_)]
        self.auswahl = {"nu": self.nu_, "m": self.m_, "val": bestes,
                        "val_alle": {f"nu={nu},m={m}": k for (nu, m), k in kennzahlen_je_paar.items()}}
        self._log(f"   Gewählt auf Validierung: nu={self.nu_}, m={self.m_}, HM {bestes['HM']}")
        return self

    # Vorhersage

    def vorhersage(self, X) -> dict:
        """Frühklassifikation ganzer Reihen, so als kämen ihre Messpunkte nacheinander an.

        Rückgabe als dict mit je einem Eintrag pro Reihe:
            klasse      Klasse, bei der die Abbruchregel entscheidet
            earliness   Präfixlänge bei der Entscheidung / Reihenlänge
            klasse_s1   Vorhersage des letzten Schnitts ohne Master, also auf der ganzen Reihe (S = 1)
        """
        geraet = slave.torch_geraet(self.geraet)
        q_je_schnitt = {}
        t_start = time.time()

        for n, (j, sl) in enumerate(sorted(self.slaves.items()), 1):
            X_praefix = praefix.stapel_praefixe(X, sl["praefix_laenge"])
            q_je_schnitt[j] = slave.slave_wahrscheinlichkeiten(sl, X_praefix, geraet)
            if n % 5 == 0 or n == len(self.slaves):
                self._log(fortschritt("Vorhersage", n, len(self.slaves), t_start))

        schritte = self._schritte(q_je_schnitt, self.masters)
        klasse, earliness = abbruch.entscheide_alle(schritte, self.laenge, self.m_)
        letzte_vorhersage = [verlauf[-1][1] for verlauf in schritte]
        return dict(klasse=[self.klassen[k] for k in klasse],
                    earliness=earliness,
                    klasse_s1=[self.klassen[k] for k in letzte_vorhersage])

    # Speichern und Laden

    def speichern(self, ordner) -> Path:
        """Schreibt slaves.joblib, masters.joblib und einstellung.json in `ordner`."""
        ordner = Path(ordner)
        ordner.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.slaves, ordner / "slaves.joblib", compress=KOMPRESSION)
        joblib.dump(self.masters, ordner / "masters.joblib", compress=KOMPRESSION)

        einstellung = dict(w=self.w_, n_kernels=self.n_kernels, C=self.C, max_iter=self.max_iter,
                           klassen=self.klassen, laenge=int(self.laenge), n_kanaele=int(self.n_kanaele),
                           nu=self.nu_, m=self.m_,
                           schnitte=[[j, sl["praefix_laenge"]] for j, sl in sorted(self.slaves.items())],
                           versionen=paket_versionen())
        (ordner / "einstellung.json").write_text(json.dumps(einstellung, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        return ordner

    @classmethod
    def laden(cls, ordner, geraet: str = "cpu", ausgabe: bool = True) -> "Fruehklassifikator":
        """Lädt ein mit speichern geschriebenes Modell."""
        ordner = Path(ordner)
        einstellung = json.loads((ordner / "einstellung.json").read_text(encoding="utf-8"))

        modell = cls(w=einstellung["w"], n_kernels=einstellung["n_kernels"], C=einstellung["C"],
                     max_iter=einstellung["max_iter"], geraet=geraet, ausgabe=ausgabe)
        modell.slaves = joblib.load(ordner / "slaves.joblib")
        modell.masters = joblib.load(ordner / "masters.joblib")
        modell.klassen = einstellung["klassen"]
        modell.laenge = einstellung["laenge"]
        modell.n_kanaele = einstellung["n_kanaele"]
        modell.w_ = einstellung["w"]
        modell.nu_ = einstellung["nu"]
        modell.m_ = einstellung["m"]
        return modell
