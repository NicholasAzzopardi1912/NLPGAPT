from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import pickle

dataset = load_dataset("MLRS/mapa_maltese")

# Function to extract SVM-style features for the current token
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

# Function that extracts the labels and features from the data given, 
# and prepares data for prediction
def prepare_data(dataset_split):
    X, y = [], []
    for item in dataset_split:
        tokens = item["tokens"]
        labels = item["level1_tags"]
        for i in range(len(tokens)):
            feats = extract_features(tokens, i)
            X.append(feats)
            y.append(labels[i])
    return X, y

X_train, y_train = prepare_data(dataset["train"])
X_test, y_test = prepare_data(dataset["test"])

# Training the SVM Model
svm_pipeline = make_pipeline(DictVectorizer(sparse=True), LinearSVC(max_iter=1000))
svm_pipeline.fit(X_train, y_train)

# Evluation of prediction results
y_pred = svm_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, digits=3, zero_division=0))

# Saving the model to reduce computation
with open("svm_for_mapa_model.pkl", "wb") as f:
    pickle.dump(svm_pipeline, f)
print("Model saved as svm_for_mapa_model.pkl")