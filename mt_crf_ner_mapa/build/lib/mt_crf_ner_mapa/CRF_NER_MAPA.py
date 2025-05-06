import pickle
from spacy.tokens import Span
import spacy
from spacy.language import Language
import os

class CRFEntityRecogniser:
    def __init__(self, model_path=None):
        # Load the pickle file containing the trained CRF model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "crf_for_mapa_model.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def extract_features(self, tokens, idx):
        # Implementing the same feature extraction logic as in the training phase
        token = tokens[idx]
        features = {
            'bias': 1.0,
            'word.lower()': token.lower(),
            'word[-3:]': token[-3:],
            'word[-2:]': token[-2:],
            'word.isupper()': token.isupper(),
            'word.istitle()': token.istitle(),
            'word.isdigit()': token.isdigit(),
        }
        if idx > 0:
            prev_token = tokens[idx - 1]
            features.update({
                '-1:word.lower()': prev_token.lower(),
                '-1:word.istitle()': prev_token.istitle(),
                '-1:word.isupper()': prev_token.isupper(),
            })
        else:
            features['BOS'] = True
        if idx < len(tokens) - 1:
            next_token = tokens[idx + 1]
            features.update({
                '+1:word.lower()': next_token.lower(),
                '+1:word.istitle()': next_token.istitle(),
                '+1:word.isupper()': next_token.isupper(),
            })
        else:
            features['EOS'] = True
        return features

    def __call__(self, doc):
        ents = []
        start = None
        end = None
        label = None


        # Extracting tokens from the doc
        tokens = []
        for token in doc:
            tokens.append(token.text)

        # Extracting features for each token
        features = []
        for idx in range(len(tokens)):
            extracted = self.extract_features(tokens, idx)
            features.append(extracted)

        X = [features]

        # Predicting tags with the CRF model
        y_pred = self.model.predict(X)[0]

        for idx, tag in enumerate(y_pred):
            if tag == 'O':
                # Outside any entity
                if start is not None and end is not None:
                    span = Span(doc, start, end, label=label)
                    ents.append(span)
                    start = end = label = None
            elif tag.startswith('B-'):
                # Beginning of a new entity
                if start is not None and end is not None:
                    # If there was a previous entity open, close it
                    span = Span(doc, start, end, label=label)
                    ents.append(span)
                
                start = idx
                end = idx + 1
                label = tag[2:]  # Remove 'B-' prefix to get the label
            elif tag.startswith('I-') and label == tag[2:]:
                # Continuation of the current entity
                end = idx + 1
            else:
                # I- without a matching B- before, treat it as B-
                if start is not None and end is not None:
                    span = Span(doc, start, end, label=label)
                    ents.append(span)
                
                start = idx
                end = idx + 1
                label = tag[2:]

        # Catch any entity left open at the end
        if start is not None and end is not None:
            span = Span(doc, start, end, label=label)
            ents.append(span)

        # Attach found entities to doc
        doc.ents = list(doc.ents) + ents
        return doc
    


@Language.factory("crf_ner_mapa")
def create_crf_ner_mapa(nlp, name):
    return CRFEntityRecogniser()
