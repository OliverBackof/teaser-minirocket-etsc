"""Schnelle Tests mit synthetischen Daten (CPU, winziges Modell). Aufruf: python -m pytest tests"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daten import teile_archiv_test  # noqa: E402
from fruehklassifikation import Fruehklassifikator, bewerte, hm  # noqa: E402
from fruehklassifikation import abbruch, master, praefix  # noqa: E402

KLASSEN = ["a", "b", "c"]
LAENGE = 80


def synthetische_reihen(n_je_klasse: int, seed: int):
    """Zwei Kanäle, Länge 80; a mit Sprung in Kanal 0, b mit Sinus in Kanal 1, c nur Rauschen."""
    rng = np.random.default_rng(seed)
    t = np.arange(LAENGE)
    X, y = [], []
    for klasse in KLASSEN:
        for _ in range(n_je_klasse):
            a = rng.normal(0, 0.3, size=(LAENGE, 2))
            if klasse == "a":
                a[t > 20, 0] += 2.0
            if klasse == "b":
                a[:, 1] += np.sin(t / 3.0)
            X.append(a.astype(np.float32))
            y.append(klasse)
    return X, y


@pytest.fixture(scope="module")
def modell():
    X, y = synthetische_reihen(10, seed=0)
    Xv, yv = synthetische_reihen(5, seed=1)
    return Fruehklassifikator(n_kernels=84, ausgabe=False).fit(X, y, Xv, yv)


def test_normierung():
    arr = np.column_stack([np.arange(20, dtype=np.float32), np.full(20, 3.0, np.float32),
                           np.r_[np.full(19, 1.0), np.nan].astype(np.float32)])
    z = praefix.normiere_praefix(arr)
    assert z.shape == (3, 20) and z.dtype == np.float32
    assert abs(float(z[0].mean())) < 1e-6 and abs(float(z[0].std()) - 1) < 1e-5
    assert np.all(z[1] == 0) and np.all(np.isfinite(z))


def test_schnitte():
    assert praefix.schnittabstand(100) == 5 and praefix.schnittabstand(10) == 1
    assert praefix.schnitte(10, 55) == [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
    assert praefix.schnitte(4, 20) == [(3, 12), (4, 16), (5, 20)]              # Präfixe unter 9 Punkten entfallen
    assert praefix.schnitte(1, 8) == [(j, j) for j in range(1, 9)]             # Reihe kürzer als 9: alle Schnitte


def test_auffuellen():
    p = np.arange(6, dtype=np.float32).reshape(3, 2)                            # 3 Punkte, 2 Kanäle
    a = praefix.auffuellen(p)
    assert a.shape == (9, 2) and np.all(a[:6] == p[0]) and np.all(a[6:] == p)
    assert praefix.stapel_praefixe([np.ones((8, 2), dtype=np.float32)], 3).shape == (1, 2, 9)


def test_abbruchregel():
    schritte = [(10, 1, True), (20, 2, True), (30, 0, True), (40, 2, True), (50, 2, True)]
    assert abbruch.entscheide_reihe(schritte, 100, 1) == (1, 0.1)
    assert abbruch.entscheide_reihe(schritte, 100, 2) == (2, 0.5)
    assert abbruch.entscheide_reihe(schritte, 100, 3) == (2, 1.0)              # ohne Entscheidung: letzter Schnitt
    unsicher = [(10, 1, False), (20, 1, True), (30, 1, False), (40, 1, True)]
    assert abbruch.entscheide_reihe(unsicher, 40, 2) == (1, 1.0)               # Unterbrechung setzt die Folge zurück


def test_kennzahlen():
    k = bewerte([0, 1, 2, 2], [0, 1, 2, 1], [0.5, 0.5, 1.0, 1.0], 3)
    assert k["Accuracy"] == 0.75 and k["Earliness"] == 0.75 and k["vorzeitig"] == 2
    assert k["HM"] == round(hm(0.75, 0.75), 4) and abs(k["F1"] - round((1 + 2 / 3 + 2 / 3) / 3, 4)) < 1e-9


def test_master_merkmale():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float32)
    ind, f = master.master_merkmale(p)
    assert list(ind) == [0, 2] and f.shape == (2, 7) and f.dtype == np.float32
    assert np.allclose(f[0], [1, 0, 0, 0.5, 0.7, 0.2, 0.1])


def test_archiv_aufteilung():
    y = ["x"] * 10 + ["y"] * 6
    i_val, i_test = teile_archiv_test(y)
    assert len(i_val) == len(i_test) == 8 and not set(i_val) & set(i_test)
    assert sum(y[i] == "y" for i in i_val) == 3 and list(i_val) == sorted(i_val)
    assert np.array_equal(teile_archiv_test(y)[0], i_val)                     # fester Seed


def test_training_und_vorhersage(modell):
    assert modell.klassen == KLASSEN and modell.w_ == 4 and modell.m_ in range(1, 6)
    assert sorted(modell.slaves) == list(range(3, 21))
    X, y = synthetische_reihen(4, seed=2)
    v = modell.vorhersage(X)
    assert len(v["klasse"]) == len(X) and set(v["klasse"]) | set(v["klasse_s1"]) <= set(KLASSEN)
    assert np.all((v["earliness"] > 0) & (v["earliness"] <= 1))
    assert np.mean(np.array(v["klasse_s1"]) == np.array(y)) > 0.8


def test_speichern_laden(modell, tmp_path):
    modell.speichern(tmp_path)
    geladen = Fruehklassifikator.laden(tmp_path, ausgabe=False)
    X, _ = synthetische_reihen(3, seed=3)
    a, b = modell.vorhersage(X), geladen.vorhersage(X)
    assert a["klasse"] == b["klasse"] and np.array_equal(a["earliness"], b["earliness"])

