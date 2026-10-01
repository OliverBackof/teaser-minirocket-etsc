"""Frühklassifikation multivariater Zeitreihen nach dem TEASER-Prinzip (MiniROCKET-Slave, OCSVM-Master).

    from fruehklassifikation import Fruehklassifikator
    modell = Fruehklassifikator().fit(X_train, y_train, X_val, y_val)
    ergebnis = modell.vorhersage(X_test)
"""
from .kennzahlen import bewerte, hm
from .modell import Fruehklassifikator

__all__ = ["Fruehklassifikator", "bewerte", "hm"]
