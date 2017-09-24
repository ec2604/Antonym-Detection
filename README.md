# Antonym-Detection
Seminar Paper - Antonym Detection

All the datasets, neural networks and code used to create and evaluate them during the writing of the seminar paper has been uploaded here in order to replicate the findings if necessary. 

# Python code details
1. The prelim module creates the word pairs and the datasets.
2. The model_utils module loads the different embedding models to memory.
3. The model_creation module creates the neural network model (keras) and trains it with the relevant datasets.
4. The model_evaluation model take an embedding model and a neural network model and evaluates their performance via Spearman's rho agains the Simlex999 dataset.

## Python package requirements:
1. Python3
2. gensim (2.3.0)
3. h5py (2.7.0)
4. Keras (1.2.0)
5. matplotlib (2.0.0)
6. nltk (3.2.2)
7. numpy (1.12.0+mkl)
8. scikit-learn (0.18.1)
9. scipy (0.18.1)
10. tensorflow (0.12.1)

