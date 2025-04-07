from datasets import load_dataset

dataset = load_dataset("unimelb-nlp/wikiann", "mt")

# Function to extract features for a token (simplified for SVM)
def word2features(sent, i):
    word = sent[i]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word[-3:]': word[-3:],  # Last 3 letters
        'word[-2:]': word[-2:],  # Last 2 letters
    }
    return features

def convert_to_svm_format(dataset):
    svm_data = []
    for item in dataset:
        tokens = item["tokens"]
        labels = item["ner_tags"]
        
        # Convert labels from int to string format
        labels = [str(label) for label in labels]
        
        # Extract features for each token and combine with label
        for i in range(len(tokens)):
            features = word2features(tokens, i)
            feature_str = " ".join([f"{k}={v}" for k, v in features.items()])
            svm_data.append(f"{feature_str} {labels[i]}")
    
    return svm_data

# Convert and save data
train_set = dataset["train"]
svm_train = convert_to_svm_format(train_set)
with open("train.svm", "w") as f:
    f.write("\n".join(svm_train))

val_set = dataset["validation"]
svm_val = convert_to_svm_format(val_set)
with open("validation.svm", "w") as f:
    f.write("\n".join(svm_val))

test_set = dataset["test"]
svm_test = convert_to_svm_format(test_set)
with open("test.svm", "w") as f:
    f.write("\n".join(svm_test))
