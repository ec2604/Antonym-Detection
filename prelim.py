from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import nltk
import pickle
import numpy as np
import csv
import os
from itertools import chain
from nltk.corpus import wordnet
from model_utils import *
from model_creation import *
from model_evaluation import *

import matplotlib.pyplot as plt


def create_synonyms(corpus_path='wacky_wiki_partial.txt', word_pair_path='./word_pairs/syns.txt'):
    """
    This method creates a word pair list, based on the parsing of a corpus.
    Antonym format: word_1,word_2,A
    Synonym format: word_1,word_2,S
    :return: None
    """
    syn_set = set()
    with open(corpus_path, 'r+') as f:
        with open(word_pair_path, 'w') as f_1:
           for line in f:
                i = 1
                print(line.lower())
                if i == 2000:
                    break
                if "<" in line or ">" in line or "." in line or  line.strip('\n').lower() in syn_set:
                    continue
                i += 1
                synonyms = wordnet.synsets(line.strip('\n').lower(), pos=wordnet.ADJ)
                syn_set.add(line.strip('\n').lower())
                lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms if word.pos() == 'a']))
                ants = set(chain.from_iterable([lemma.lemmas()[0].antonyms() for lemma in synonyms]))
                ant_words = set([lemma.name() for lemma in ants])
                for word in lemmas:
                    if word.lower() == line.strip('\n').lower():
                        continue
                    print("{0},{1},S\n".format(line.strip('\n').lower(), word.lower()))
                    f_1.write("{0},{1},S\n".format(line.strip('\n').lower(), word))
                for word in ant_words:
                    print("{0},{1},A\n".format(line.strip('\n').lower(), word.lower()))
                    f_1.write("{0},{1},A\n".format(line.strip('\n').lower(), word))



def create_cooccurrence_vectors(base_dir, corpus_path='wacky_wiki_corpus_en1.words'):
    """
    This function creates raw co_occurrence vectors from a given corpus.
    :param base_dir: The directory to save the raw co-occurrence vectors to.
    :param corpus_path: The path of the corpus to parse
    :return: None
    """
    os.makedirs(base_dir)
    with open(corpus_path, 'r+') as f:
        #pair_vocab contains all the distinct words in the word_pair list
        with open('./word_pairs/pair_vocab.txt', 'r+') as vocab_list:
            print("Creating vocab list...")
            vocab_set = set(vocab_list.read().splitlines())
            print("Done creating vocab list!")
            print("Splitting document...")
            doc = f.read().splitlines()
            print("Done splitting document!")
            k=2
            print("Producing n-grams...")
            my_ngrams = [" ".join(ngram) for ngram in nltk.ngrams(doc, 2*k+1) if ngram[k].lower() in vocab_set]
            print("Done producing n-grams!")
            my_additional_stop_words = ['<s>', '</s>']
            stop_words_full = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
            count_model = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words_full)
            print("Performing fit_transform...")
            X = count_model.fit_transform(my_ngrams)
            #Save the name of each feature in the co-occurrence vectors
            np.save('./word_pairs/feature_names.npy', np.array(count_model.get_feature_names()))
            #Saves the dictionary for later use
            with open('./word_pairs/vocab_dictionary', 'wb') as f_1:
                pickle.dump(count_model.vocabulary_, f_1)
            print("Done performing fit_transform!")
            print("Performing matrix multiplication...")
            Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
            print("Done performing matrix multiplication!")
            Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
            print("Saving words...")
            for word in vocab_set:
                print("Saving co-occurence vector for {0}".format(word))
                idx = count_model.vocabulary_.get(word)
                if idx is None:
                    continue
                curr_vec = np.array(Xc[idx,:].todense()).astype(np.float16)
                np.save("{0}/{1}".format(base_dir, word), curr_vec)
            print("Done saving words")



def create_glove_data_sets(model, target_dir='./glove_data_sets', word_pair_path='./word_pairs/syns.csv'):
    """
    This function creates the data sets from a GloVe model as input for a neural network by stacking the embeddings on top of
    each other.
    :param model: The model used for the original vector embedding
    :param target_dir: The directory to which to save the stacked embeddings
    :param word_pair_path: The path of the csv containing the word pairs
    :return: None
    """
    with open(word_pair_path, 'r+') as f:
        reader = csv.reader(f)
        os.makedirs(target_dir)
        for row in reader:
            curr_vec_name = "{0}/{1}_{2}_{3}".format(target_dir, row[0], row[1], row[2])
            try:
                if not row[0] in model.keys() or row[1] not in model.keys():
                    print("Skipping {0}-{1} pair".format(row[0], row[1]))
                    continue
                np.save(curr_vec_name, np.vstack((model[row[0]], model[row[1]])))
            except Exception as e:
                print(e)
                print("Could not create data set for {0} vs. {1}".format(row[0], row[1]))
                continue


def create_ono_data_sets(model, target_dir='./glove_data_sets_ono', synonym_file='./word_pairs/syns_ono.csv',
                         antonym_file='./word_pairs/ant_ono.csv'):
    """
    This function creates the data sets from a GloVe model as input for a neural network by stacking the embeddings on top of
    each other. Different from create_glove_data_sets in that it uses two different files for parsing synonyms and antonyms.
    :param model: The model used for the original vector embedding
    :param target_dir: The directory to which to save the stacked embeddings
    :param word_pair_path: The path of the csv containing the word pairs
    :return: None
    """
    with open(synonym_file, 'r+') as f:
        reader = csv.reader(f)
        os.makedirs(target_dir)
        for row in reader:
            if len(row) == 0:
                continue
            if not row[0] in model.keys():
                continue
            for i in range(1,len(row)):
                curr_vec_name = "{0}/{1}_{2}_{3}".format(target_dir, row[0], row[i], 'S')
                try:
                    if row[1] not in model.keys():
                        print("Skipping {0}-{1} pair".format(row[0], row[1]))
                        continue
                    np.save(curr_vec_name, np.vstack((model[row[0]], model[row[1]])))

                except Exception as e:
                    print(e)
                    print("Could not create data set for {0} vs. {1}".format(row[0], row[1]))
                    continue
    with open(antonym_file, 'r+') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            if not row[0] in model.keys():
                continue
            for i in range(1,len(row)):
                curr_vec_name = "{0}/{1}_{2}_{3}".format(target_dir, row[0], row[i], 'A')
                try:
                    if row[1] not in model.keys():
                        print("Skipping {0}-{1} pair".format(row[0], row[1]))
                        continue
                    np.save(curr_vec_name, np.vstack((model[row[0]], model[row[1]])))

                except Exception as e:
                    print(e)
                    print("Could not create data set for {0} vs. {1}".format(row[0], row[1]))
                    continue

def create_gensim_data_sets(model, target_dir='./gensim_data_sets', word_pair_file_path='./word_pairs/syns.csv'):
    """
    This function creates the data sets from a gensim model as input for a neural network by stacking the embeddings on top
    of each other.
    :param model: The VSM model to use for the original vector embeddings
    :param target_dir: The directory to which to save the stacked vector embeddings
    :param word_pair_file_path: The word pair file path
    :return:
    """
    with open(word_pair_file_path, 'r+') as f:
        reader = csv.reader(f)
        os.makedirs(target_dir)
        for row in reader:
            curr_vec_name = "{0}/{1}_{2}_{3}".format(target_dir, row[0], row[1], row[2])
            try:
                if not row[0] in model.vocab or row[1] not in model.vocab:
                    print("Skipping {0}-{1} pair".format(row[0], row[1]))
                    continue
                np.save(curr_vec_name, np.vstack((model[row[0]], model[row[1]])))
            except Exception as e:
                print(e)
                print("Could not create data set for {0} vs. {1}".format(row[0], row[1]))
                continue


def create_simlex_pair_idx_exclusion_list(data_set_dir, target_file_name,
                                          simlex_words_file_path='./SimLex-999/words_only.csv'):
    """
    This function creates an np array containing all the indices of word pairs that exist in Simlex999 and the dataset. This
    files is used in order to avoid training the neural network on potential validation pairs.
    :param data_set_dir: The directory of the data set
    :param target_file_name: The file name of the npy file to be saved
    :param simlex_words_file_path: The csv containing the Simlex999 pairs
    :return: None
    """
    files = os.listdir(data_set_dir)
    with open(simlex_words_file_path, 'r') as f:
        sim_words = f.readlines()
    sim_words = [x.strip('\n') for x in sim_words]
    ex_list = []
    for word in sim_words:
        pair_1, pair_2 = word[:word.find(',', -5)].split(',')
        if pair_1 + "_" + pair_2 + "_A.npy" in files:
            ex_list.append(files.index(pair_1 + "_" + pair_2 + "_A.npy"))
        elif pair_2 + "_" + pair_1 + "_A.npy" in files:
            ex_list.append(files.index(pair_2 + "_" + pair_1 + "_A.npy"))
        elif pair_1 + "_" + pair_2 + "_S.npy" in files:
            ex_list.append(files.index(pair_1 + "_" + pair_2 + "_S.npy"))
        elif pair_2 + "_" + pair_1 + "_S.npy" in files:
            ex_list.append(files.index(pair_2 + "_" + pair_1 + "_S.npy"))
    np.save(target_file_name, np.array(ex_list))

def create_simlex_data_sets(embedding_model, target_dir, containment_func, simlex_words='./SimLex-999/words_only.csv'):
    """
    This function takes the VSM and creates the relevant simlex vectors for it in the specified directory
    :param embedding_model: The embedding model
    :param target_dir: The target directory
    :return: None
    """
    with open(simlex_words, 'r+') as f:
        reader = csv.reader(f)
        os.makedirs(target_dir)
        for row in reader:
            try:
                curr_vec_name = "{0}/{1}_{2}".format(target_dir, row[0], row[1])
                if not containment_func(embedding_model, row[0]) or not containment_func(embedding_model, row[1]):
                    print("Skipping {0}-{1} pair".format(row[0], row[1]))
                    continue
                np.save(curr_vec_name, np.vstack((embedding_model[row[0]], embedding_model[row[1]])))
            except Exception as e:
                print(e)
                print("Could not create data set for {0} vs. {1}".format(row[0], row[1]))
                continue

if __name__ == '__main__':
    pass