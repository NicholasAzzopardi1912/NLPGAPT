from datasets import load_dataset
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load train set from preprocessed data
with open("ner_as_svm_trainset.pkl", "rb") as f:
    train_svm = pickle.load(f)


# Load test set from preprocessed data
with open("ner_as_svm_testset.pkl", "rb") as f:
    test_svm = pickle.load(f)



# Convert list of tuples into separate lists
def prepare_data(svm_dataset, max_len=20):  # Set a fixed max length
    X = [features for _, features in svm_dataset]  # Extract features
    y = [label for label, _ in svm_dataset]  # Extract labels

    # Pad all feature vectors to max_len (default 20)
    X_padded = pad_sequences(X, maxlen=max_len, padding="post", dtype="float32")

    # Encode labels into numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_padded, y_encoded, label_encoder

X_train, y_train, label_encoder = prepare_data(train_svm, max_len=20)
X_test, y_test, _ = prepare_data(test_svm, max_len=20)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train SVM Model
svm_model = SVC(kernel="linear", class_weight="balanced")  # Standard SVM with class balancing
svm_model.fit(X_train, y_train_encoded)


# Evaulations on test set
y_test_pred = svm_model.predict(X_test)
y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

print("\nTest Set Performance:")
print(classification_report(y_test, y_test_pred_labels))
