import spacy
import SVM_NER_MAPA

# Loading the saved SVM NER pipeline
nlp = spacy.load("./mt_SVM_NER_MAPA")

# Test with a sample sentence
doc = nlp("Maria kienet il-Prim Ministru ta' Malta.")

# Printing found entities
if doc.ents:
    for ent in doc.ents:
        print(ent.text, ent.label_)
else:
    print("No entities found.")