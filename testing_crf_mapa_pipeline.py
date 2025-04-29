import spacy
import CRF_NER_MAPA

# Loading the saved CRF NER pipeline
nlp = spacy.load("./mt_CRF_NER_MAPA")

# Test with a sample sentence
doc = nlp("Maria kienet il-Prim Ministru ta' Malta.")

# Printing found entities
if doc.ents:
    for ent in doc.ents:
        print(ent.text, ent.label_)
else:
    print("No entities found.")