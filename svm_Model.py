import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the preprocessed data (as produced by svm_preprocessing.py)
with open("svm_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
label_encoder = data["label_encoder"]

# Define and train the SVM model
svm_model = SVC(kernel="linear", class_weight="balanced")

print("Training SVM model...")
svm_model.fit(X_train, y_train)
print("Training complete.")

# Save the trained SVM model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
print("SVM model saved as svm_model.pkl.")

# Predict and evaluate on the test set
y_pred = svm_model.predict(X_test)
target_names = [str(cls) for cls in label_encoder.classes_]

print("\nSVM - Test Set Performance:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
