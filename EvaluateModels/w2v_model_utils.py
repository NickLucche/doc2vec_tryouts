import gensim
from gensim.models import Word2Vec
import random
import numpy as np
import re
import unicodedata

def online_training(model, training_corpus, epochs = 5):
    """
        model: word2vec (pre-trained) model to update.
        training_corpus: training corpus on which to update the model.
        epochs: number of epochs for which to update the model; low 
        values of epochs is advised, in order not to change already learned vectors.
        
        Update a w2v model (teach meaning of unknown words to model), by re-training 
        the model over the new sentences.
        Returns trained model.
    """
    # re-train model using abstract AND entities
    print("Lenght before update", len(model.wv.vocab))
    
    random.shuffle(training_corpus)
    model.build_vocab(training_corpus, update=True)

    # choose a low epochs value 
    model.train(training_corpus, total_examples=model.corpus_count, epochs=epochs)

    print("Lenght after update", len(model.wv.vocab))
    
    return model
    # model.save()
    
def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)
    
## TODO: update with WEIGHTED MEAN OF VECTORS
def mean_of_vectors(vectors, vec_size):
    """given a list of vectors (and their size), return the simplest mean of vectors."""

    ## vectors might be empty if no entity is recognized
    if len(vectors) < 1:
        print('Unknown doc found')
        return []
    # all word vectors have the same shape (always the same number of features)
    sum_vectors = np.zeros((vec_size,), dtype="float32")
    for vec in vectors:
        sum_vectors = np.add(sum_vectors, vec)
    sum_vectors = np.divide(sum_vectors, len(vectors))
    return sum_vectors

def infer_vector(entities, model, index2word_set, verbose = False):
    """Given a list of entities, returns the vector representing the document from which the entities 
    were extracted from, wrt a given W2V model.
    
    entities: list of entities, our way of representing a single document.
    model: w2v model.
    
    Returns (vector) representation of given document.
    """
    unknown_words = 0
    # get word vector of each entity; ignores word if the model does not know it
    entities_vecs = []
    unknown = []
    
    for e in entities:
        e = strip_accents(e)  # strip the accents from the word
        
        if e in index2word_set: 
            entities_vecs.append(model[e])
        elif e.lower() in index2word_set:
            entities_vecs.append(model[e.lower()])
        else:
            # unknown word, try to split the word (maybe be formed by multiple words)
            # and use the sum of single-word vectors as the vector of the entity
            words = e.split()
            word_v = np.zeros((model.wv.vector_size, ), dtype="float32")
            u = 0
            for word in words:
                if word in index2word_set:
                    word_v = np.add(word_v, model[word])
                elif word.lower() in index2word_set:
                    word_v = np.add(word_v, model[word.lower()])
                else: # keep trying, get an approximation for it at least
                    u += 1
            if u==len(words):
                unknown.append(e)
                unknown_words += 1
            else: # at least one sub-word was recognized
                entities_vecs.append( np.divide(word_v, len(word_v)) )
    if verbose:
        percentage = (len(entities)-unknown_words) * 100 / len(entities)
        print("Known words: {}%  ({}/{})".format(percentage, (len(entities)-unknown_words), len(entities)))
        if len(unknown) > 0: print("Unknown entities found: ", unknown)
    return mean_of_vectors(entities_vecs, model.wv.vector_size)
    