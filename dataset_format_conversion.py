from datasets import load_dataset
import spacy
from spacy.tokens import DocBin
#from spacy_transformers import TransformersLanguage

dataset = load_dataset("unimelb-nlp/wikiann", "mt")

#test_dataset = dataset["test"]
#print(test_dataset[0])

print(dataset["train"].features["ner_tags"])

label_map = {0: "O", 1: "PER", 2: "PER", 3: "ORG", 4: "ORG", 5: "LOC", 6: "LOC"}

# Function that takes a dataset split, converts each entry to spacy format, and returns a list of tuples
def dataset_to_spacy_format_conversion(split):
    spacy_data = []

# Creating full sentences 
    for i in split:
        tokens = i["tokens"]
        ner_tags = i["ner_tags"]
        text = " ".join(tokens)

        entities = []
        start = 0

# Converting the tokens to their spans based off of the number of characters
        for token, tag in zip(tokens, ner_tags):
            end = start + len(token)
            if tag != 0:
                entity_label = label_map[tag]
                entities.append((start, end, entity_label))
            start = end + 1

        spacy_data.append((text, {"entities": entities}))

    return spacy_data    

# Converting each split of the dataset
train_data = dataset_to_spacy_format_conversion(dataset["train"])
valid_data = dataset_to_spacy_format_conversion(dataset["validation"])
test_data = dataset_to_spacy_format_conversion(dataset["test"])

print(train_data[:3])


# Saving converted dataset to .spacy file format
def save_as_spacy_format(data, output_path):
    # An empty SpaCy pipeline for Maltese is created to act as a placeholder for the language whilst working on it
    nlp = spacy.blank("xx")  
    doc_bin = DocBin()

    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        
        for begin, end, label in annotations["entities"]:
            span = doc.char_span(begin, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        
        doc.ents = ents
        doc_bin.add(doc)

    doc_bin.to_disk(output_path)
    print(f"Saved: {output_path}")

# Save train, validation, and test datasets
save_as_spacy_format(train_data, "train.spacy")
save_as_spacy_format(valid_data, "valid.spacy")
save_as_spacy_format(test_data, "test.spacy")