# Antonym-Detection - Seminar Paper

The purpose of the seminar paper was to attempt to show the possibility of discerning between antnonyms and synonyms based on their semantical distributional differences and to use these results to better existing VSM models' predictions.

All the datasets, neural networks and code used to create and evaluate them during the writing of the seminar paper has been uploaded here in order to replicate the findings if necessary. 

# Creating datasets and evaluating model

To replicate the findings or re-use the code, one needs to:
1. Have a word pair list ready (or use the one in the word pairs folder). 
2. Have a vector embedding model ready (e.g GloVE or word2vec), these can be loaded using the model_utils module.
3. Create the datasets using the prelim module.
4. Create and train the neural network with the model_creation module.
5. Evaluate the neural network model and the VSM (vector space model) using the model_evaluation module. 

# Python code details

## Overview of modules

1. The prelim module creates the word pairs and the datasets.
2. The model_utils module loads the different embedding models to memory.
3. The model_creation module creates the neural network model (keras) and trains it with the relevant datasets.
4. The model_evaluation model take an embedding model and a neural network model and evaluates their performance via Spearman's rho agains the Simlex999 dataset.

## Python package requirements

1. Python3
2. gensim (2.3.0)
3. h5py (2.7.0)
4. Keras (1.2.0) - backend configuration file uploaded.
5. matplotlib (2.0.0)
6. nltk (3.2.2)
7. numpy (1.12.0+mkl)
8. scikit-learn (0.18.1)
9. scipy (0.18.1)
10. tensorflow (0.12.1)

