import spacy
import  CRF_NER_MAPA

# Creating a blank Maltese pipeline
nlp = spacy.blank("xx")

# Adding the CRF NER component to the pipeline
nlp.add_pipe("crf_ner_mapa")

# Saving pipeline to disk
nlp.to_disk("./mt_CRF_NER_MAPA")

print("Pipeline saved at ./mt_CRF_NER_MAPA")