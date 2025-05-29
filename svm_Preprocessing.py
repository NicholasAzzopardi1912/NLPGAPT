from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the WikiANN dataset (Maltese)
dataset = load_dataset("unimelb-nlp/wikiann", "mt")

# Function to extract features for a token
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
    }

    # Previous word features
    if i > 0:
        prev = sent[i-1]
        features.update({
            '-1:word.lower()': prev.lower(),
            '-1:word.istitle()': prev.istitle(),
            '-1:word.isupper()': prev.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    # Next word features
    if i < len(sent) - 1:
        nxt = sent[i+1]
        features.update({
            '+1:word.lower()': nxt.lower(),
            '+1:word.istitle()': nxt.istitle(),
            '+1:word.isupper()': nxt.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence

    return features

def convert_dataset_to_svm(dataset_split):
    """Convert HuggingFace dataset split to SVM-ready format."""
    X = []
    y = []
    for example in dataset_split:
        tokens = example["tokens"]
        labels = example["ner_tags"]
        for i in range(len(tokens)):
            features = word2features(tokens, i)
            X.append(features)
            y.append(labels[i])
    return X, y

# --- Preprocess train and test data ---

X_train_dict, y_train = convert_dataset_to_svm(dataset["train"])
X_test_dict, y_test = convert_dataset_to_svm(dataset["test"])

# Vectorize feature dicts
vectorizer = DictVectorizer(sparse=False)
X_train = vectorizer.fit_transform(X_train_dict)
X_test = vectorizer.transform(X_test_dict)

# Standard scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# --- Save preprocessed data for use in the model script ---
with open("svm_preprocessed.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_encoded,
        "y_test": y_test_encoded,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }, f)
