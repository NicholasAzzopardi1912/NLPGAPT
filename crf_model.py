from sklearn_crfsuite.metrics import flat_classification_report
import pickle
from sklearn_crfsuite import CRF

# Load the training and test sets
with open("ner_as_crf_trainset.pkl", "rb") as f:
    train_crf = pickle.load(f)
with open("ner_as_crf_testset.pkl", "rb") as f:
    test_crf = pickle.load(f)

X_train = [[token[1] for token in sentence] for sentence in train_crf]
y_train = [[token[2] for token in sentence] for sentence in train_crf]
X_test = [[token[1] for token in sentence] for sentence in test_crf]
y_test = [[token[2] for token in sentence] for sentence in test_crf]

# Creating the CRF Model
crf = CRF(algorithm='lbfgs', c1=0.01, c2=0.01, max_iterations=100, all_possible_transitions=True)

print("CRF Model is training:")
crf.fit(X_train, y_train)
print("Training has finished")

# Pickle is being used to be able to train the model once and then save it. Reducing computation and time
with open("crf_model.pkl", "wb") as f:
    pickle.dump(crf, f)
print("Model saved as crf_model.pkl")

# Evaluate the model
y_pred = crf.predict(X_test)
print(flat_classification_report(y_test, y_pred))