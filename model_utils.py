import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from model_creation import create_model

def load_glove_model(glove_file):
    """
    This function loads the glove model found at the specified file
    :param glove_file: The path of the pretrained-model
    :return: the model
    """
    print("Loading Glove Model...")
    f = open(glove_file,'r+', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


def load_gensim_model(gensimFile, binary_mode=True, limit=False, limit_range=0):
    """
    This function loads the gensim model found at the specified file
    :param glove_file: The path of the pretrained-model
    :return: the model
    """
    print("Loading gensim model...")
    if limit:
        model = gensim.models.KeyedVectors.load_word2vec_format(gensimFile, binary=binary_mode, limit=limit_range)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(gensimFile, binary=binary_mode)
    print("Done loading model!")
    return model

def load_nn_model(input_dim, weights_file):
    """
    Loads the neural network with the desired input dimensions and weights file
    :param input_dim: The dimensions of the stacked input vectors (2x input_dim)
    :param weights_file: The keras weights file of the pre-trained model
    :return:
    """
    model = create_model(input_dim)
    model.load_weights(weights_file)
    return model