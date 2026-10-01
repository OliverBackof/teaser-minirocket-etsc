"""Slave: der Klassifikator an einem Schnitt.

Für jeden Schnitt j gibt es einen eigenen Slave. Er bekommt die Präfixe aller Reihen bis zu diesem Schnitt und
schätzt für jede Reihe, wie wahrscheinlich jede Klasse ist. Ein Slave besteht aus drei Stufen:

    Präfix -> MiniROCKET -> StandardScaler    -> logistische Regression -> Wahrscheinlichkeiten q
              (Merkmale)    (Standardisierung)   (Klassifikation)

Gespeichert wird ein Slave als dict:

    praefix_laenge   Länge s_j des Präfixes in Messpunkten
    minirocket       MiniROCKET-Objekt aus tsai (MiniRocketFeatures), liegt immer auf der CPU
    skalierer        StandardScaler
    logreg           LogisticRegression
    klassen          Reihenfolge der Klassen, in der die Spalten von q stehen
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Einstellungen

# Startwert des Zufallsgenerators von MiniROCKET (bestimmt, welche Reihen für die Biases gezogen werden)
MINIROCKET_SEED = 42

# Wie viele Reihen MiniROCKET auf einmal verarbeitet. Nur eine Frage des Speichers, das Ergebnis ist gleich.
REIHEN_JE_BLOCK = 64


# Gerät (CPU oder GPU)

def torch_geraet(geraet: str):
    """Wandelt "cpu" oder "gpu" in ein torch-Gerät um.

    Ist keine CUDA-fähige GPU vorhanden, wird mit einer Warnung auf die CPU ausgewichen.
    """
    import torch

    if geraet == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if geraet == "gpu":
        print("WARNUNG: keine CUDA-GPU gefunden, MiniROCKET läuft auf der CPU", flush=True)
    return torch.device("cpu")


# Stufe 1: MiniROCKET

def erzeuge_minirocket(n_kanaele: int, n_punkte: int, n_merkmale: int, geraet):
    """Legt ein neues, noch nicht angepasstes MiniROCKET-Objekt für Präfixe dieser Form an."""
    from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures

    return MiniRocketFeatures(c_in=n_kanaele, seq_len=n_punkte, num_features=n_merkmale,
                              random_state=MINIROCKET_SEED).to(geraet)


def passe_minirocket_an(minirocket, X: np.ndarray) -> None:
    """Bestimmt die Biases von MiniROCKET aus den Trainingsreihen X (Reihen x Kanäle x Punkte).

    MiniROCKET faltet jede Reihe mit festen Kernen. Aus welchem Wert der Faltung ein Merkmal wird, legen die
    Biases fest. Sie sind Quantile der Faltungsergebnisse einzelner Trainingsreihen.

    Das Ergebnis ist identisch zu MiniRocketFeatures.fit(X) aus tsai, braucht aber viel weniger Speicher:
    tsai faltet dort bis zu num_dilations x 84 Reihen, verwendet für die Biases aber nur 84 davon (eine je Kern).
    Hier werden mit denselben Zufallszahlen genau diese 84 Reihen gezogen und nur sie gefaltet.
    """
    import torch
    import torch.nn.functional as F

    n_reihen = X.shape[0]
    n_kerne = minirocket.num_kernels
    geraet = minirocket.kernels.device

    # Dieselben zwei Ziehungen wie in tsai: erst die Reihen für fit(), daraus dann eine Reihe je Kern
    n_gezogen = min(n_reihen, minirocket.num_dilations * n_kerne)
    np.random.seed(minirocket.random_state)
    gezogen = np.random.choice(n_reihen, n_gezogen, False)
    np.random.seed(minirocket.random_state)
    reihe_je_kern = gezogen[np.random.choice(n_gezogen, n_kerne)]

    reihen = torch.from_numpy(np.ascontiguousarray(X[reihe_je_kern])).to(geraet)
    kern_nr = torch.arange(n_kerne, device=geraet)

    with torch.no_grad():
        for i, (dilatation, rand) in enumerate(zip(minirocket.dilations, minirocket.padding)):
            # Jede gezogene Reihe mit allen Kernen falten ...
            faltung = F.conv1d(reihen, minirocket.kernels, padding=rand, dilation=dilatation,
                               groups=minirocket.c_in)
            # ... die Kanäle so kombinieren, wie MiniROCKET es für diese Dilatation vorsieht
            faltung = faltung.reshape(n_kerne, minirocket.c_in, n_kerne, -1)
            kombination = getattr(minirocket, f"channel_combinations_{i}")
            faltung = torch.mul(faltung, kombination).sum(1)
            # ... und für Kern k nur das Ergebnis der Reihe k behalten
            proben = faltung[kern_nr, kern_nr]

            # Die Biases sind Quantile dieser Werte
            quantile = minirocket._get_quantiles(minirocket.num_features_per_dilation[i]).to(geraet)
            setattr(minirocket, f"biases_{i}", torch.quantile(proben, quantile, dim=1).T)
            del faltung

    minirocket.prefit = torch.BoolTensor([True])


def minirocket_merkmale(minirocket, X: np.ndarray, geraet) -> np.ndarray:
    """Berechnet die MiniROCKET-Merkmale für die Präfixe X (Reihen x Kanäle x Punkte).

    Rückgabe: Matrix Reihen x Merkmale. Nach einer Rechnung auf der GPU wird das MiniROCKET-Objekt wieder auf die
    CPU gelegt und der GPU-Speicher freigegeben, damit er bei vielen Schnitten nacheinander nicht vollläuft.
    """
    import torch
    from tsai.models.MINIROCKET_Pytorch import get_minirocket_features

    auf_gpu = geraet.type == "cuda"
    minirocket.to(geraet)
    merkmale = get_minirocket_features(X, minirocket, chunksize=REIHEN_JE_BLOCK, use_cuda=auf_gpu, to_np=True)
    merkmale = merkmale.squeeze(-1)          # letzte Achse hat Länge 1

    if auf_gpu:
        minirocket.cpu()
        torch.cuda.empty_cache()
    return merkmale


# Stufe 3: Wahrscheinlichkeiten der logistischen Regression

def wahrscheinlichkeiten(logreg, merkmale: np.ndarray, klassen: list) -> np.ndarray:
    """Klassenwahrscheinlichkeiten q (Reihen x Klassen, float32) in der Reihenfolge `klassen`.

    Die logistische Regression kennt nur die Klassen, die an diesem Schnitt im Training vorkamen. Fehlt eine Klasse,
    bekommt sie die Wahrscheinlichkeit 0, damit q an jedem Schnitt dieselben Spalten hat.
    """
    spalte_der_klasse = {klasse: i for i, klasse in enumerate(klassen)}
    q_logreg = logreg.predict_proba(merkmale)

    q = np.zeros((merkmale.shape[0], len(klassen)), dtype=np.float32)
    for i, klasse in enumerate(logreg.classes_):
        q[:, spalte_der_klasse[klasse]] = q_logreg[:, i]
    return q


# Slave trainieren und anwenden

def trainiere_slave(X: np.ndarray, y: list, klassen: list, praefix_laenge: int, n_kernels: int, C: float,
                    max_iter: int, geraet):
    """Trainiert einen Slave auf den normierten Präfixen X (Reihen x Kanäle x Punkte) mit den Labels y.

    Rückgabe: (Slave-dict, Wahrscheinlichkeiten der Trainingsreihen). Die Wahrscheinlichkeiten auf den eigenen
    Trainingsreihen braucht der Master, er wird auf den richtig klassifizierten davon trainiert.
    """
    # Stufe 1: MiniROCKET an die Trainingsreihen anpassen und Merkmale berechnen
    minirocket = erzeuge_minirocket(n_kanaele=X.shape[1], n_punkte=X.shape[2], n_merkmale=n_kernels, geraet=geraet)
    passe_minirocket_an(minirocket, X)
    merkmale = minirocket_merkmale(minirocket, X, geraet)

    # Stufe 2: Merkmale standardisieren
    skalierer = StandardScaler()
    merkmale = skalierer.fit_transform(merkmale).astype(np.float32, copy=False)

    # Stufe 3: logistische Regression, jede Klasse gleich gewichtet (class_weight="balanced")
    logreg = LogisticRegression(C=float(C), max_iter=max_iter, class_weight="balanced", solver="lbfgs")
    logreg.fit(merkmale, y)

    slave = dict(praefix_laenge=int(praefix_laenge), minirocket=minirocket.cpu(), skalierer=skalierer,
                 logreg=logreg, klassen=list(klassen))
    return slave, wahrscheinlichkeiten(logreg, merkmale, klassen)


def slave_wahrscheinlichkeiten(slave: dict, X: np.ndarray, geraet) -> np.ndarray:
    """Wendet einen trainierten Slave auf die normierten Präfixe X an.

    Rückgabe: Wahrscheinlichkeiten q (Reihen x Klassen, float32).
    """
    merkmale = minirocket_merkmale(slave["minirocket"], X, geraet)
    merkmale = slave["skalierer"].transform(merkmale).astype(np.float32, copy=False)
    return wahrscheinlichkeiten(slave["logreg"], merkmale, slave["klassen"])
