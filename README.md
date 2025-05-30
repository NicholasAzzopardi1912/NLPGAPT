The following README.md file outlines the name and purpose of the files:

Packaging Process:
  The Folder mt_crf_ner_mapa consists of the files required to package the crf MAPA model into installable files for the user
  The Folder mt_svm_ner_mapa consists of the files required to package the svm MAPA model into installable files for the user


Pipeline and Package Testing:
  testing_crf_mapa_pipeline.py file was used to test the pipeline and packaging process for CRF
  testing_svm_mapa_pipeline.py file was used to test the pipeline and packaging process for SVM


Wikiann Models:
The CRF Model is made up of multiple files:
    crf_Preprocessing.py contains the preprocessing performed on the wikiann dataset for CRF and saves the preprocessed data to
      ner_as_crf_trainset.pkl and ner_as_crf_testset.pkl
    crf_model.py uses the ner_as_crf_trainset.pkl and ner_as_crf_testset.pkl pickle files to then train and test the model
      These results are then stored in the crf_model.pkl

The SVM Model is made up of multiple files:
    svm_Preprocessing.py contains the preprocessing performed on the wikiann dataset for SVM and saves the preprocessed data to
      svm_preprocessed.pkl
    svm_Model.py uses the svm_preprocessed.pkl file to then train and test the model
      These results are then stored in the svm_model.pkl


MAPA Models:
The CRF Model along with the construction of the pipeline is made up of the following:
  CRF_for_mapa_maltese.py consists of the preprocessing and training of the model. These results are then stored in
    crf_for_mapa_model.pkl, found in the mt_crf_ner_mapa folder
  CRF_NER_MAPA.py takes the results stored in crf_for_mapa_model.pkl, and extracts the results for the found entities in 'doc'
  That python file is then imported by CRF_Pipeline.py where the pipeline is created and stored on disk

The SVM Model along with the construction of the pipeline is made up of the following:
  SVM_for_mapa_maltese.py consists of the preprocessing and training of the model. These results are then stored in
    svm_for_mapa_model.pkl, found in the mt_svm_ner_mapa folder
  SVM_NER_MAPA.py takes the results stored in svm_for_mapa_model.pkl, and extracts the results for the found entities in 'doc'
  That python file is then imported by SVM_Pipeline.py where the pipeline is created and stored on disk
  

    
