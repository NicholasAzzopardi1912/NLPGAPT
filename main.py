from datasets import load_dataset
import spacy
from spacy.tokens import DocBin


dataset = load_dataset("unimelb-nlp/wikiann", "mt")
test_dataset = dataset["test"]

print(test_dataset[0])

print("hello")