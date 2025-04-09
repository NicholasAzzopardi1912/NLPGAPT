from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

dataset = load_dataset("unimelb-nlp/wikiann", "mt")

# Function to extract features for a token
def word2features(sent, i):

    # Extracting features for the current word in the sentence
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
    # Applying features from the previous word 
    if i > 0:
        prev = sent[i-1]
        features.update({
            '-1:word.lower()': prev.lower(),
            '-1:word.istitle()': prev.istitle(),
            '-1:word.isupper()': prev.isupper(),
        })
    else:
        features['BOS'] = True  # Speacial case for beginning of sentence

    # Adding features from the next word
    if i < len(sent) - 1:
        nxt = sent[i+1]
        features.update({
            '+1:word.lower()': nxt.lower(),
            '+1:word.istitle()': nxt.istitle(),
            '+1:word.isupper()': nxt.isupper(),
        })
    else:
        features['EOS'] = True  # Speacial case for end of sentence

    return features

def convert_dataset_to_svm(dataset):
    
    # Lists storing all feature dictionaries and corresponding NER labels
    X = []
    y = []

    # For each sentence in the dataset, convert each token into a feature dictionary and pair with its corresponding label
    for example in dataset:
        tokens = example["tokens"]
        labels = example["ner_tags"]

        for i in range(len(tokens)):
            features = word2features(tokens, i)
            X.append(features)
            y.append(labels[i])

    return X, y

# Preprocessing is applied to the train and test sets to a sutiable format for SVM
X_train_dict, y_train = convert_dataset_to_svm(dataset["train"])
X_test_dict, y_test = convert_dataset_to_svm(dataset["test"])

# Vectorising the feature dictionaries into a numerical format
vectorizer = DictVectorizer(sparse=False)
X_train = vectorizer.fit_transform(X_train_dict)
X_test = vectorizer.transform(X_test_dict)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#Train the SVM model
svm_model = SVC(kernel="linear", class_weight="balanced")
svm_model.fit(X_train_scaled, y_train_encoded)

#Predict and evaluate
y_pred = svm_model.predict(X_test_scaled)
target_names = [str(cls) for cls in label_encoder.classes_]

print("\nTest Set Performance:")
print(classification_report(y_test_encoded, y_pred, target_names = target_names, zero_division=0))
