# mt_crf_ner_mapa

A spaCy-compatible Named Entity Recognition (NER) component for the Maltese language, powered by a CRF (Conditional Random Field) model trained on the [MLRS MAPA dataset](https://huggingface.co/datasets/MLRS/mapa_maltese).

## Features

- Custom CRF-based NER integrated into the spaCy pipeline
- Trained on the `level1_tags` from the MAPA Maltese dataset
- Plug-and-play spaCy component (`crf_ner_mapa`)

## Installation

Build and install the package:

```bash
python setup.py sdist bdist_wheel
pip install dist/mt_crf_ner_mapa-1.0.0-py3-none-any.whl
