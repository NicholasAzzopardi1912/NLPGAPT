from datasets import load_dataset
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import pickle

dataset = load_dataset("MLRS/mapa_maltese")

# Function to extract CRF-style features for the current token
def extract_features(tokens, idx):
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

    # Adding features from the previous word
    if idx > 0:
        prev_token = tokens[idx - 1]
        features.update({
            '-1:word.lower()': prev_token.lower(),
            '-1:word.istitle()': prev_token.istitle(),
            '-1:word.isupper()': prev_token.isupper(),
        })
    else:
        features['BOS'] = True

    # Adding features from the next word
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

# Function to convert the dataset into CRF format
def convert_to_crf_format(dataset_split):
    crf_data = []

    for item in dataset_split:
        tokens = item["tokens"]
        labels = item["level1_tags"]
        sentence = [(tokens[i], extract_features(tokens, i), labels[i]) for i in range(len(tokens))]
        crf_data.append(sentence)

    return crf_data

# Apply conversion function
train_data = convert_to_crf_format(dataset["train"])
test_data = convert_to_crf_format(dataset["test"])

# Splitting the data into features and labels
X_train = [[token[1] for token in sent] for sent in train_data]
y_train = [[token[2] for token in sent] for sent in train_data]
X_test = [[token[1] for token in sent] for sent in test_data]
y_test = [[token[2] for token in sent] for sent in test_data]


# Training the CRF model
crf = CRF( algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

crf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = crf.predict(X_test)

print(metrics.flat_classification_report(y_test, y_pred, digits=3, zero_division=0))

# Saving the model to reduce computation
with open("crf_for_mapa_model.pkl", "wb") as f:
    pickle.dump(crf, f)
print("Model saved as crf_for_mapa_model.pkl")