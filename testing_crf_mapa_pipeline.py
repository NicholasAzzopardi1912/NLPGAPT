import spacy
import mt_crf_ner_mapa
import mt_crf_ner_mapa.CRF_NER_MAPA as crf_mod

# Confirming that the test is correctly using the CRF_NER_MAPA module from inside the mt_crf_ner_mapa package
print(">>> USING FILE:", crf_mod.__file__)

nlp = spacy.blank("xx")
nlp.add_pipe("crf_ner_mapa")

# Test with a sample sentence
doc = nlp("Maria kienet il-Prim Ministru ta' Malta.")

# Printing found entities
if doc.ents:
    for ent in doc.ents:
        print(ent.text, ent.label_)
else:
    print("No entities found.")