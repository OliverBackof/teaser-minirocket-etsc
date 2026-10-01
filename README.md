# Frühklassifikation auf UEA-Datensätzen

Code zur Bachelorarbeit. Er zeigt das Verfahren der Arbeit an öffentlichen Datensätzen des UEA-Archivs:
Frühklassifikation multivariater Zeitreihen nach dem TEASER-Prinzip mit einem MiniROCKET-Slave und einem
One-Class-SVM-Master. Das Modell sieht die Messpunkte einer Reihe nacheinander und entscheidet so früh wie möglich,
zu welcher Klasse sie gehört.

## Schnellstart

Getestet mit Python 3.14 (Windows 11, CPU). Für Python 3.14 wird numba ab Version 0.63 gebraucht.

```bash
pip install -r requirements.txt
python experiment.py --datensaetze BasicMotions      # etwa 1 Minute, lädt die Daten beim ersten Mal herunter
python experiment.py                                 # sieben Beispieldatensätze, etwa 20 Minuten auf der CPU
```

Die Arbeit wertet 26 Datensätze aus. Die übrigen lassen sich über `--datensaetze` rechnen (siehe unten).

Die Ergebnisse erscheinen als Tabelle auf der Konsole und liegen danach im Ordner `ergebnisse/`.

## Das Verfahren

```
Reihe:      x1 x2 x3 ... ------------------------------------------------>  Zeit
Schnitte:            s_1 = w        s_2 = 2w        s_3 = 3w      ...      (w = Reihenlänge / 20)

an jedem Schnitt j:
    Präfix -> Slave j -> Wahrscheinlichkeiten q -> Master j -> verlässlich ja / nein

Abbruchregel:
    m verlässliche Vorhersagen derselben Klasse in Folge -> Entscheidung
```

- **Slave:** Das Präfix der ersten s_j Messpunkte wird je Kanal normiert und von MiniROCKET in 5000 Merkmale
  übersetzt. Eine logistische Regression schätzt daraus die Wahrscheinlichkeit jeder Klasse.
- **Master:** Eine One-Class-SVM prüft, ob die Ausgabe des Slaves so aussieht wie bei richtig klassifizierten
  Trainingsreihen. Nur dann gilt die Vorhersage als verlässlich.
- **Abbruchregel:** Sobald m verlässliche Vorhersagen derselben Klasse aufeinander folgen, ist die Reihe
  entschieden. Kommt es nicht dazu, gilt die Vorhersage am Ende der Reihe.
- **Wahl der Einstellungen:** nu (Parameter der One-Class-SVM) und m werden auf den Validierungsdaten gewählt,
  und zwar das Paar mit dem höchsten HM. Danach wird genau einmal auf den Testdaten ausgewertet.

**Aufteilung der Daten:** Train ist der Trainingsteil des Archivs. Der Testteil des Archivs wird je Klasse zur
Hälfte in Validierung und Test geteilt.

## Kennzahlen

| Kennzahl | Bedeutung |
|---|---|
| S=1 Acc | Accuracy auf der ganzen Reihe, ohne Frühabbruch |
| Frühabbruch Acc | Accuracy der Klasse, bei der die Abbruchregel entscheidet |
| Earliness | mittlerer Anteil der Reihe, der bis zur Entscheidung gesehen wurde (kleiner = früher) |
| HM | harmonisches Mittel aus Accuracy und 1 - Earliness |
| F1 | Macro-F1 über alle Klassen beim Frühabbruch |

## Aufruf

Aus diesem Ordner heraus:

```bash
python experiment.py                                        # die sieben Beispieldatensätze
python experiment.py --datensaetze BasicMotions Epilepsy    # bestimmte Datensätze
python experiment.py --geraet gpu                           # MiniROCKET auf der GPU
python experiment.py --speichern                            # trainierte Modelle zusätzlich speichern
```

| Option | Bedeutung |
|---|---|
| `--datensaetze` | Namen der UEA-Datensätze |
| `--geraet` | `cpu` (Standard) oder `gpu` |
| `--daten` | Download-Ordner, Standard `uea_daten/` |
| `--ergebnisse` | Ergebnisordner, Standard `ergebnisse/` |
| `--speichern` | Modelle unter `ergebnisse/modelle/<Datensatz>/` speichern |

## Ausgaben

- `ergebnisse/<Datensatz>.json`: alle Kennzahlen, gewähltes nu und m und die Entscheidung je Testreihe
- `ergebnisse/uebersicht.md`: Tabelle aller gerechneten Datensätze

## Andere UEA-Datensätze

Jeder andere multivariate Datensatz des UEA-Archivs mit gleich langen Reihen wird nur über seinen Namen gewählt:

```bash
python experiment.py --datensaetze Cricket RacketSports UWaveGestureLibrary
```

Die Daten werden beim ersten Aufruf heruntergeladen. Große Datensätze wie EigenWorms oder PEMS-SF rechnet man
besser mit `--geraet gpu`. Bei Reihen, die kürzer als die 9 Punkte langen MiniROCKET-Kerne sind (PenDigits mit 8
Punkten), werden die Präfixe links mit dem ersten Messwert aufgefüllt.

## Aufbau

```
experiment.py          Hauptskript
daten.py               UEA-Datensatz laden und aufteilen
fruehklassifikation/   das Modell
    praefix.py         Schnitte und Normierung der Präfixe
    slave.py           MiniROCKET und logistische Regression
    master.py          One-Class-SVM
    abbruch.py         Abbruchregel
    kennzahlen.py      Accuracy, Earliness, HM, F1
    modell.py          Fruehklassifikator: Training, Vorhersage, Speichern, Laden
tests/                 kurze Tests mit künstlichen Daten (python -m pytest tests)
```

## Bezug zur Arbeit

Verfahren und Einstellungen sind dieselben wie in der Arbeit. In der Arbeit wurde allerdings ein vertraulicher Datensatz
zu Prüfläufen von Batterien verwendet, deswegen wurde der Ansatz hier auf öffentlich verfügbare Datensätze zugeschnitten.
In dieser Implementierung wurde nur die symmetrische Politik behandelt.
