# Transformer Encoder für Executable Classification

Dieses Repository enthält die Implementierung eines Transformer Encoders zur Klassifizierung von Executables. Diese Implementierung wurde im Rahmen der Bachelorarbeit "Entwicklung und Evaluation eines Transformer-basierten Ansatzes zur Klassifizierung von Executables" entwickelt.

## Detaillierte Informationen

Detaillierte Informationen zur Modellarchitektur, zum verwendeten Datensatz, Performance-Metriken und Evaluationsergebnisse finden Sie in der zugehörigen Bachelorarbeit "Entwicklung und Evaluation eines Transformer-basierten Ansatzes zur Klassifizierung von Executables".

## Voraussetzungen

- Python >= 3.12
- Weights & Biases Account (für Hyperparameter-Optimierung)

## Installation

1. Virtual Environment erstellen und aktivieren:
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

## Projektstruktur

Diese Struktur bildet nur den wichtigsten Teil ab:

```
├── src/
│   ├── training/
│   │   ├── *.py
│   │   └── *.ipynb
│   └── evaluation/
│       ├── eval_model_100.ipynb
│       ├── eval_model_200.ipynb
│       └── eval_model_300.ipynb
├── models/
└── outputs/
```

## Training

### Einzelmodell-Training

1. Öffnen Sie die gewünschte Training-Pipeline-Datei (`.py` oder `.ipynb`) im `src/training/` Verzeichnis
2. Passen Sie die `fixed_config` mit den gewünschten Parametern an
3. Starten Sie das Training:
```bash
python3 src/training/training_pipeline_v5_single_innovator.py  # Für das Innovator-Modell
```

### Hyperparameter-Optimierung

1. Stellen Sie sicher, dass Sie einen Weights & Biases Account haben und eingeloggt sind
2. Passen Sie die `sweep_configuration` nach Ihren Bedürfnissen an
3. Starten Sie die Optimierung für die gewünschte Sequenzlänge:
```bash
python3 src/training/training_pipeline_v5_200.py  # Für Sequenzlänge 200
```

## Modelle und Dateien

- Die fertig trainierten Modelle werden automatisch im Ordner `models/` gespeichert
- Die Evaluationsmetriken für jedes Modell werden im Ordner `outputs/` gespeichert

## Evaluation

Zur Evaluation eines trainierten Modells und Generierung von Metriken für Testdaten:

1. Wählen Sie das entsprechende Notebook basierend auf der Sequenzlänge:
   - `src/evaluation/eval_model_100.ipynb` für Modelle mit Sequenzlänge 100
   - `src/evaluation/eval_model_200.ipynb` für Modelle mit Sequenzlänge 200
   - `src/evaluation/eval_model_300.ipynb` für Modelle mit Sequenzlänge 300

## Hardware-Empfehlungen

- min 12GB auf der GPU empfohlen
- Das Training benötigt eine CUDA-fähige GPU

