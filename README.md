ML for NLP: Coursework 1 Task 2
    This repository contains the full implementation for Task 2 of the ‘Machine Learning for Natural Langauge Processing’ coursework. 
The system performs multi class text classification on a subset of the 20 Newsgroups dataset, comparing several feature configurations based on TD_IDF (unigrams, bigrams and document length) and selecting the best model using a development set.
The code is fully documented and can be run end to end from the terminal.

Repository Contents:
1. main.py                 # Main experiment script (train/dev/test pipeline)
2. functions.py            # Preprocessing, feature extraction and model utilities
3. 20_Newsgroups.csv       # Dataset used in the experiments (text + label)
4. requirements.txt        # Python dependencies
5. README.md               # Documentation (this file)


The program performs the following steps:
1. Load and preprocess the dataset
    Clean text (remove headers, emails, URLs, punctuation)
    Convert to lowercase
    Compute document length (word count)
2. Split the data into Train / Development / Test
    80% training
    10% development
    10% test
    (stratified to preserve class balance)
3. Build and evaluate multiple feature configurations
    Unigrams TF–IDF
    Bigrams TF–IDF
    Document length
    All possible combinations of the above
4. Train a Linear SVM classifier for each configuration.
5. Select the best model based on development macro-F1.
6. Run 5-fold cross-validation on the training set for the best model.
7. Retrain the best model on Train+Dev and produce final test accuracy, macro-precision, macro-recall, and macro-F1.

All results are printed directly in the terminal.

How to run the programme:
1. Create a virtual envionment
While this step is not strictly required it is highly recommended as you will need to install 3 external python packages required for running the programme. If you do not use a virtual environment, these packages will be installed into your global Python environment instead. 

Windows:
python -m venv venv
venv\Scripts\activate

Linux and Mac:
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Run the Programme
python main.py

Modifying feature configurations:
Document length can take a longer time to run than unigram and bigram, if you want to run the code at any point without any specific configuration of features you can do this by commenting out the specific entry inside configs list found in main.py.

Side Notes:
- All code is fully commented
- The required dataset is included in the directory
- The random seed has been fixed at '42'. If you change this it may affect reproducability.

Contact: If you have any issues please contact me at 'jonesel32@cardiff.ac.uk'
I hope the code works as intended and you have a good day! :>