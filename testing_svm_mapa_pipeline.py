import spacy
import mt_svm_ner_mapa
import mt_svm_ner_mapa.SVM_NER_MAPA as svm_mod

# Confirming that the test is correctly using the SVM_NER_MAPA module from inside the mt_svm_ner_mapa package
print(">>> USING FILE:", svm_mod.__file__)

nlp = spacy.blank("xx")
nlp.add_pipe("svm_ner_mapa")

# Run a test sentence
doc = nlp("Maria kienet il-Prim Ministru ta' Malta.")

# Print entities
if doc.ents:
    for ent in doc.ents:
        print(ent.text, ent.label_)
else:
    print("No entities found.")