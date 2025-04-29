import spacy
import SVM_NER_MAPA

# Creating a blank Maltese pipeline
nlp = spacy.blank("xx")

# Adding the SVM NER component to the pipeline
nlp.add_pipe("svm_ner_mapa")

# Saving pipeline to disk
nlp.to_disk("./mt_SVM_NER_MAPA")

print("Pipeline saved at ./mt_SVM_NER_MAPA")
