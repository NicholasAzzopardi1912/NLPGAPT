from datasets import load_dataset
import pickle
import numpy as np

dataset = load_dataset("unimelb-nlp/wikiann", "mt")

# Function to extract features for a token
def word2features(sent, i):
    word = sent[i]
    features = [
        1.0 if word.isupper() else 0.0,  # 'word.isupper()'
        1.0 if word.istitle() else 0.0,  # 'word.istitle()'
        1.0 if word.isdigit() else 0.0,  # 'word.isdigit()'
    ]
    
    # Convert last 3 and 2 letters to one-hot encodings
    features += [1.0 if c == word[-3:] else 0.0 for c in set(''.join(sent[i-1:i+2]).lower())]
    features += [1.0 if c == word[-2:] else 0.0 for c in set(''.join(sent[i-1:i+2]).lower())]
    
    # Add features for previous and next words
    if i > 0:
        features.append(1.0)  # 'BOS'
        features += [1.0 if c == sent[i-1].lower() else 0.0 for c in set(''.join(sent[i-2:i]).lower())]
    else:
        features.append(0.0)
    
    if i < len(sent) - 1:
        features.append(0.0)  # 'EOS'
        features += [1.0 if c == sent[i+1].lower() else 0.0 for c in set(''.join(sent[i:i+2]).lower())]
    else:
        features.append(1.0)
    
    return features

def convert_wikiann_to_svm(dataset):
    svm_data = []

    for item in dataset:
        tokens = item["tokens"]
        labels = item["ner_tags"]
        
        # Convert labels from int to string
        labels = [dataset.features["ner_tags"].feature.int2str(label) for label in labels]
        
        # Generate features per token and convert to SVM format
        svm_sent = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            features = word2features(tokens, i)
            svm_sent.append((label, features))
        svm_data.extend(svm_sent)
    
    return svm_data

# Converting train dataset to be applicable for SVM
svm_trainset = convert_wikiann_to_svm(dataset["train"])

# Saving the train set in a text file
with open("ner_as_svm_trainset.txt", "w") as f:
    for label, features in svm_trainset:
        feature_str = " ".join([f"{i+1}:{v}" for i, v in enumerate(features) if v != 0.0])
        f.write(f"{label} {feature_str}\n")

# Converting validation dataset to be applicable for SVM
svm_validset = convert_wikiann_to_svm(dataset["validation"])

# Saving the validation set in a text file
with open("ner_as_svm_validset.txt", "w") as f:
    for label, features in svm_validset:
        feature_str = " ".join([f"{i+1}:{v}" for i, v in enumerate(features) if v != 0.0])
        f.write(f"{label} {feature_str}\n")

# Converting test dataset to be applicable for SVM
svm_testset = convert_wikiann_to_svm(dataset["test"])

# Saving the test set in a text file
with open("ner_as_svm_testset.txt", "w") as f:
    for label, features in svm_testset:
        feature_str = " ".join([f"{i+1}:{v}" for i, v in enumerate(features) if v != 0.0])
        f.write(f"{label} {feature_str}\n")

# Saving the preprocessed data to pickle files
with open("ner_as_svm_trainset.pkl", "wb") as f:
    pickle.dump(svm_trainset, f)

with open("ner_as_svm_validset.pkl", "wb") as f:
    pickle.dump(svm_validset, f)

with open("ner_as_svm_testset.pkl", "wb") as f:
    pickle.dump(svm_testset, f)

# Print some data to verify
print("\nSample SVM formatted train data (first 5 samples):")
for label, features in svm_trainset[:5]:
    feature_str = " ".join([f"{i+1}:{v}" for i, v in enumerate(features) if v != 0.0])
    print(f"{label} {feature_str}")

print("\nAll tests passed successfully!")
