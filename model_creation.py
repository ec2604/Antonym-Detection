import os
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.layers import Dense, Flatten, normalization
from keras.models import Sequential
from keras.layers import Dropout



def create_model(embedding_dimension):
    """
    This function creates the neural network architecture, and compiles it.
    :return: The compiled model
    """
    model = Sequential()
    model.add(Dense(10, input_shape=(2, embedding_dimension), init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(5, init='normal', activation='relu'))
    model.add(normalization.BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(2, init='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def training_set(train_set_idx, base_dir):
    """
    This function returns a generator that feeds the neural network training batches of 100.
    :param train_set_idx: The indices to be used for training batches.
    :param base_dir: The directory of the dataset.
    :return: Generator yielding batches
    """
    files = np.array(os.listdir(base_dir))
    while True:
        np.random.shuffle(train_set_idx)
        x = [np.load('{0}/{1}'.format(base_dir,example)) for example in files[train_set_idx[0:100]]]
        y = [np.array([1,0]) if os.path.splitext(fname)[0].split('_')[2] == "A" else np.array([0,1]) for fname in files[train_set_idx[0:100]]]
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        yield (x,y)


def validation_set(val_set_idx, base_dir):
    """
    This function returns a generator that feeds the neural network training batches of 100.
    :param val_set_idx: The indices to be used for validation batches.
    :param base_dir: The directory of the dataset.
    :return: Generator yielding batches
    """
    files = np.array(os.listdir(base_dir))
    while True:
        np.random.shuffle(val_set_idx)
        x = [np.load('{0}/{1}'.format(base_dir, example)) for example in files[val_set_idx[0:100]]]
        y = [np.array([1,0]) if os.path.splitext(fname)[0].split('_')[2] == "A" else np.array([0,1]) for fname in files[val_set_idx[0:100]]]
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        yield (x,y)


def round_nearest_hundred_below(num):
    """
    This utility function rounds to the nearest hundred from below
    :param num: Number to round
    :return: Rounded number
    """
    return int(math.ceil(num / 100.0)) * 100


def train_nn_model(base_dir, input_dim, target_weights_file, exclusion_list, train_ratio,
                   num_epoch, show_metric_plots=True):
    """
    This function creates a neural network and trains it using data sets created before hand. The weights are saved to
    a specified file for further use.
    :param base_dir: The directory of the datasets
    :param input_dim: The input dimension of the neural network, needs to be compatible with the datasets
    :param target_weights_file: The path of the weights file to be saved
    :param exclusion_list: The exclusion list (should be created beforehand) to be used to avoid training on simlex
    validation files.
    :param train_ratio: The split between training and validation
    :param num_epoch: Number of epochs to run network
    :param show_metric_plots: Whether to show graphs of the accuracy or not.
    :return: The trained model
    """
    NUM_SIMLEX_FILES = 999
    model = create_model(input_dim)
    num_files = len(os.listdir(base_dir)) - NUM_SIMLEX_FILES
    train_set_bound = round_nearest_hundred_below(num_files*float(train_ratio))
    ex_list = np.load(exclusion_list)
    idx = np.setdiff1d(np.arange(num_files), ex_list)
    np.random.shuffle(idx)
    train_set = idx[0:train_set_bound]
    val_set = idx[train_set_bound:]
    val_set = np.concatenate([val_set, ex_list])
    vset = validation_set(val_set, base_dir)
    tset = training_set(train_set, base_dir)
    history = model.fit_generator(tset, samples_per_epoch=train_set_bound,
                 nb_epoch=num_epoch, validation_data=vset,
                 nb_val_samples=round_nearest_hundred_below(val_set.size), verbose=1)
    model.save_weights(target_weights_file)
    if show_metric_plots:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    return model