from setuptools import setup, find_packages

setup(
    name="mt_svm_ner_mapa",
    version="1.0.0",
    description="SVM-based NER for Maltese integrated with spaCy",
    author="Nicholas Azzopardi, Jamie Bugeja, Matthew Borg Barthet",
    author_email="nicholas.azzopardi.23@um.edu.mt, jamie.bugeja.23@um.edu.mt, matthew.borg-barthet.23@um.edu.mt",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "spacy>=3.8.5",
        "scikit-learn==1.6.1",
        "numpy==2.2.5",
        "sklearn-crfsuite",
        "datasets"
    ],
    python_requires="==3.12.9",
    entry_points={
        "spacy_factories": [
            "svm_ner_mapa = mt_svm_ner_mapa.SVM_NER_MAPA:create_svm_ner_mapa"
        ]
    }
)