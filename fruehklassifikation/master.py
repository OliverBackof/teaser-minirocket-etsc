"""Master: entscheidet an einem Schnitt, ob die Vorhersage des Slaves verlässlich ist.

Für jeden Schnitt j gibt es einen eigenen Master, eine One-Class-SVM (OCSVM). Sie lernt, wie die Ausgaben des
Slaves bei richtig klassifizierten Trainingsreihen typischerweise aussehen. Liegt eine neue Ausgabe innerhalb
dieses Bereichs, gilt sie als verlässlich.

Eingabe des Masters je Reihe (Master-Merkmale):

    [ vorhergesagte Klasse als One-hot | Abstand der zwei höchsten Wahrscheinlichkeiten | Wahrscheinlichkeiten q ]

Gespeichert wird ein Master als dict:

    ocsvm    OneClassSVM, oder None bei zu wenigen Beispielen (dann ist keine Ausgabe verlässlich)
    gamma    gewähltes gamma des RBF-Kerns
    nu       nu der OCSVM
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM

# Einstellungen

# Kandidaten für gamma. Großes gamma = enge Grenze um die Beispiele, kleines gamma = weite Grenze.
GAMMA_GITTER = [100, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1]

# Höchstens so viele Beispiele gehen in eine OCSVM ein (Rechenzeit), zufällig gezogen mit festem Startwert
MAX_BEISPIELE = 5000
SEED = 42

# Mit weniger richtig klassifizierten Reihen wird an diesem Schnitt kein Master trainiert
MIN_BEISPIELE = 3


# Master-Merkmale

def master_merkmale(q: np.ndarray):
    """Bildet aus den Slave-Wahrscheinlichkeiten q (Reihen x Klassen) die Eingabe des Masters.

    Rückgabe: (vorhergesagte Klasse je Reihe als Index, Master-Merkmale je Reihe).
    """
    reihe = np.arange(len(q))
    rangfolge = np.argsort(-q, axis=1)
    vorhergesagt = rangfolge[:, 0]
    zweitbeste = rangfolge[:, 1]

    abstand = q[reihe, vorhergesagt] - q[reihe, zweitbeste]
    one_hot = np.zeros((len(q), q.shape[1]), dtype=np.float32)
    one_hot[reihe, vorhergesagt] = 1.0

    merkmale = np.concatenate([one_hot, abstand[:, None], q], axis=1)
    return vorhergesagt, merkmale


# Master trainieren und anwenden

def trainiere_master(q: np.ndarray, y_idx: np.ndarray, nu: float) -> dict:
    """Trainiert den Master eines Schnitts.

    q sind die Wahrscheinlichkeiten des Slaves auf seinen eigenen Trainingsreihen, y_idx deren wahre Klassen als
    Index. Nur richtig klassifizierte Reihen gehen ein. gamma wird per Kreuzvalidierung gewählt: bewertet wird,
    wie viele der zurückgehaltenen richtig klassifizierten Beispiele die OCSVM einschließt.
    """
    vorhergesagt, merkmale = master_merkmale(q)
    merkmale = merkmale[vorhergesagt == np.asarray(y_idx)]

    if len(merkmale) < MIN_BEISPIELE:
        return dict(ocsvm=None, gamma=None, nu=float(nu))
    if len(merkmale) > MAX_BEISPIELE:
        zufall = np.random.default_rng(SEED)
        merkmale = merkmale[zufall.choice(len(merkmale), size=MAX_BEISPIELE, replace=False)]

    # Alle Beispiele sind "eingeschlossen" (+1), die Accuracy ist damit der Anteil, den die OCSVM einschließt
    suche = GridSearchCV(OneClassSVM(kernel="rbf", nu=nu), {"gamma": GAMMA_GITTER}, scoring="accuracy",
                         cv=min(10, len(merkmale)), n_jobs=1)
    suche.fit(merkmale, np.ones(len(merkmale)))

    ocsvm = suche.best_estimator_
    return dict(ocsvm=ocsvm, gamma=float(ocsvm.gamma), nu=float(nu))


def verlaesslich(master: dict, q: np.ndarray) -> np.ndarray:
    """True je Reihe, wenn der Master die Slave-Ausgabe als verlässlich einstuft.

    Ohne OCSVM (zu wenige Trainingsbeispiele) ist keine Ausgabe verlässlich.
    """
    if master["ocsvm"] is None:
        return np.zeros(len(q), dtype=bool)
    _, merkmale = master_merkmale(q)
    return master["ocsvm"].predict(merkmale) == 1
