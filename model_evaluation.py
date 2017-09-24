import csv
import os
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from numpy.linalg import lstsq
from sklearn.model_selection import KFold
#The antonym-synonym detection model evaluation module

def cosine_similarity(embedding_model, word_1, word_2):
    """
    This function computes a simple cosine similarity between vectors.
    :param embedding_model: The embedding model from which to take the word embeddings
    :param word_1: The key of the first word in the embedding_model data structure
    :param word_2: The key of the second word in the embedding_model data structure
    :return: The cosine similarity
    """
    cosine_sim = np.dot(embedding_model[word_1], embedding_model[word_2]).astype(np.float32) / (np.linalg.norm(embedding_model[word_1]).astype(np.float32) *
                                                                          np.linalg.norm(embedding_model[word_2]).astype(np.float32))
    return cosine_sim


def neural_network_baseline(base_dir, nn_model):
    """
    This function calculates the spearman correlation of the neural network model against the whole Simlex999 dataset.
    :param base_dir: The base directory containing the embeddings (vector embeddings used to train the model)
     of the Simlex999 word pairs.
    :param nn_model: The trained neural network model used to make the predictions
    :return: Spearman correlation of the model's predictions against the whole Simlex999 dataset.
    :return:
    """
    with open('./SimLex-999/words_only.csv', 'r') as csvfile:
        pair_reader = csv.reader(csvfile, delimiter=',')
        pred_list = []
        sim_list = []
        for row in pair_reader:
            if Path(base_dir + row[0] + "_" + row[1] + ".npy").exists():
                curr_pair = np.load(base_dir+ row[0] + "_" + row[1] + ".npy")
            elif Path(base_dir + row[1] + "_" + row[0] + ".npy").exists():
                curr_pair = np.load(base_dir + row[1] + "_" + row[0] + ".npy")
            else:
                continue
            pred_list.append(nn_model.predict(curr_pair[np.newaxis,...])[0][1])
            sim_list.append(row[2])
        print("NN Model against all of simlex: {0}".format(spearmanr(pred_list, sim_list)[0]))


def embedding_model_baseline(base_dir, embedding_model):
    """
    This function calculates the spearman correlation of the embedding model against the whole Simlex999 dataset.
    :param base_dir: The base directory containing the embeddings of the Simlex999 word pairs.
    :param embedding_model: The embedding
    :return: Spearman correlation of the model's prediction against the whole Simlex999 dataset.
    """
    with open('./SimLex-999/words_only.csv', 'r') as simlex_f:
        pair_reader = csv.reader(simlex_f, delimiter=',')
        pred_list = []
        sim_list = []
        for row in pair_reader:
            if not Path(base_dir + row[0] + "_" + row[1] + ".npy").exists() and not Path(base_dir + row[1] + "_" + row[0] + ".npy").exists():
                continue
            pred_list.append(cosine_similarity(embedding_model, row[0], row[1]))
            sim_list.append(row[2])
        print(len(pred_list))
        print("Spearman rho for embedding model against simlex: {0}".format(spearmanr(pred_list, sim_list)[0]))


def fold_validation(base_dir, embedding_model, nn_model, train_set, val_set):
    """
    This  function takes a training set and a validation set and uses OLS (least squares) a to try to fit a
    linear combination (with intercept) between the VSM model and the nueral network. The spearman correlation is
    calculated against the validation set.
    :param base_dir: The base directory containing the embedding of the Simlex999 vectors.
    :param embedding_model: The model
    :param nn_model: The trained neural network model used to classify word pairs.
    :param train_set: A list of indices to be used as the training set (indices of the files list used in this function).
    :param val_set: A list of indices to be used as the validation set (indices of the files list used in this function).
    :return: The spearman correlation against the validation set.
    """
    files = np.array(os.listdir(base_dir))
    pred_list = []
    model_res = []
    sim_list = []
    with open ('./SimLex-999/words_only.csv', 'r') as simlex_words_f:
        pair_reader = csv.reader(simlex_words_f, delimiter=',')
        for row in pair_reader:
            first_pair =  row[0] + "_" + row[1] + ".npy"
            second_pair = row[1] + "_" + row[0] + ".npy"
            if not (first_pair in files[train_set] or second_pair in files[train_set]):
                continue
            if Path(base_dir+ row[0] + "_" + row[1] + ".npy").exists():
                curr_pair = np.load(base_dir+ row[0] + "_" + row[1] + ".npy")
            elif Path(base_dir+ row[1] + "_" + row[0] + ".npy").exists():
                curr_pair = np.load(base_dir + row[1] + "_" + row[0] + ".npy")
            else:
                continue
            pred_list.append(nn_model.predict(curr_pair[np.newaxis,...])[0][1])
            model_res.append(cosine_similarity(embedding_model, row[0], row[1]))
            sim_list.append(row[2])
        x = np.vstack([np.array(pred_list), np.array(model_res), np.ones(len(model_res))])
        coeff = lstsq(np.transpose(x), sim_list)[0]
        model_res = []
        pred_list = []
        sim_list = []
        with open ('./SimLex-999/words_only.csv', 'r') as simlex_words_f_2:
            pair_reader = csv.reader(simlex_words_f_2, delimiter=',')
            for row in pair_reader:
                first_pair =  row[0] + "_" + row[1] + ".npy"
                second_pair = row[1] + "_" + row[0] + ".npy"
                if not (first_pair in files[val_set] or second_pair in files[val_set]):
                    continue
                if Path(base_dir+ row[0] + "_" + row[1] + ".npy").exists():
                    curr_pair = np.load(base_dir+ row[0] + "_" + row[1] + ".npy")
                elif Path(base_dir+ row[1] + "_" + row[0] + ".npy").exists():
                    curr_pair = np.load(base_dir + row[1] + "_" + row[0] + ".npy")
                else:
                    continue
                pred_list.append(nn_model.predict(curr_pair[np.newaxis,...])[0][1])
                model_res.append(cosine_similarity(embedding_model, row[0], row[1]))
                sim_list.append(row[2])
            x = np.transpose([np.array(pred_list), np.array(model_res), np.ones(len(model_res))])
            res = np.dot(x, coeff)
        return spearmanr(np.transpose(res), sim_list)[0]


def model_combination_rho(base_dir, embedding_model, nn_model, k):
    """
    This function takes the VSM model and the neural network model and uses k-fold cross validation to approximate
    the spearman's rho achievable by model combination.
    :param base_dir: The base directory for the simlex vectors used for the evaluation
    :param embedding_model: The VSM with which the neural network is combined
    :param nn_model: The neural network (keras based) model used for prediction
    :param k: The number of splits for k-fold cross validation
    :return: The average spearman's rho with the Simlex999 dataset.
    """
    simlex_set = np.arange(998)
    kf = KFold(n_splits=k)
    res = []
    for train_index, val_index in kf.split(simlex_set):
        res.append(fold_validation(base_dir, embedding_model, nn_model, train_index, val_index))
    return np.mean(np.array(res))