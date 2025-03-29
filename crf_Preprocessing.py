from datasets import load_dataset
import pickle

dataset = load_dataset("unimelb-nlp/wikiann", "mt")

# Function to extract CRF-style features for a token
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word[-3:]': word[-3:],  # Last 3 letters
        'word[-2:]': word[-2:],  # Last 2 letters
    }
    
    if i > 0:
        features.update({
            '-1:word.lower()': sent[i-1].lower(),
        })
    else:
        features['BOS'] = True  # Beginning of Sentence

    if i < len(sent) - 1:
        features.update({
            '+1:word.lower()': sent[i+1].lower(),
        })
    else:
        features['EOS'] = True  # End of Sentence

    return features



def convert_wikiann_to_crf(dataset):
    crf_data = []

    for item in dataset:
        tokens = item["tokens"]
        labels = item["ner_tags"]
        
        # Convert labels from int to string
        labels = [dataset.features["ner_tags"].feature.int2str(label) for label in labels]
        
        # Generate features per token
        crf_sent = [(token, word2features(tokens, i), labels[i]) for i, token in enumerate(tokens)]
        crf_data.append(crf_sent)
    
    return crf_data

# Converting train dataset to be applicable for crf
crf_trainset = convert_wikiann_to_crf(dataset["train"])

# Saving the train set in a text file
with open("ner_as_crf_trainset.txt", "w") as f:
    for sent in crf_trainset:
        for word, features, label in sent:
            feature_str = " ".join([f"{k}={v}" for k, v in features.items()])
            f.write(f"{word} {feature_str} {label}\n")
        f.write("\n")

# Converting validation dataset to be applicable for crf
crf_validset = convert_wikiann_to_crf(dataset["validation"])

# Saving the validation set in a text file
with open("ner_as_crf_validset.txt", "w") as f:
    for sent in crf_validset:
        for word, features, label in sent:
            feature_str = " ".join([f"{k}={v}" for k, v in features.items()])
            f.write(f"{word} {feature_str} {label}\n")
        f.write("\n")


# Converting test dataset to be applicable for crf
crf_testset = convert_wikiann_to_crf(dataset["test"])

# Saving the test set in a text file
with open("ner_as_crf_testset.txt", "w") as f:
    for sent in crf_testset:
        for word, features, label in sent:
            feature_str = " ".join([f"{k}={v}" for k, v in features.items()])
            f.write(f"{word} {feature_str} {label}\n")
        f.write("\n")


# Saving the preprocessed data to pickle files
with open("ner_as_crf_trainset.pkl", "wb") as f:
    pickle.dump(crf_trainset, f)
with open("ner_as_crf_testset.pkl", "wb") as f:
    pickle.dump(crf_testset, f)








# Test 1: Check if the output is a list
assert isinstance(crf_trainset, list), "Output is not a list"

# Test 2: Check if each element in the output list is a list (sentence representation)
assert all(isinstance(sent, list) for sent in crf_trainset), "Not all sentences are lists"

# Test 3: Check if each token entry is a tuple (word, features, label)
assert all(isinstance(entry, tuple) and len(entry) == 3 for sent in crf_trainset for entry in sent), "Some token entries are not tuples of (word, features, label)"

# Test 4: Check if feature extraction is working
first_sentence = crf_trainset[0] if crf_trainset else []
if first_sentence:
    first_token = first_sentence[0]
    assert isinstance(first_token[1], dict), "Feature set is not a dictionary"

# Test 5: Print sample output
print("Sample processed sentence (first 3 tokens):")
for token_entry in first_sentence[:3]:
    print(token_entry)

print("\nAll tests passed successfully!")