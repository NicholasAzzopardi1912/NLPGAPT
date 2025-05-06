# mt_svm_ner_mapa

A spaCy-compatible Named Entity Recognition (NER) component for the Maltese language, powered by a Support Vector Machine (SVM) classifier trained on the [MLRS MAPA dataset](https://huggingface.co/datasets/MLRS/mapa_maltese).

This package provides a custom spaCy pipeline component, `svm_ner_mapa`, which can be added to any spaCy pipeline and used to extract entities from Maltese text.

## 📦 Installation

First, make sure your Python environment has the necessary dependencies:

```bash
python setup.py sdist bdist_wheel
pip install mt_svm_ner_mapa-1.0.0-py3-none-any.whl
