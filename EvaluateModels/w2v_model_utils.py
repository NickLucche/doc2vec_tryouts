import gensim
from gensim.models import Word2Vec
import random
import numpy as np

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
    
## TODO: update with WEIGHTED MEAN OF VECTORS
def mean_of_vectors(vectors):
    """given a list of vectors, return the simplest mean of vectors."""
    ## vectors might be empty if no entity is recognized
    if vectors == []:
        print('Unknown doc found')
        return []
    sum_vectors = np.zeros(np.shape(vectors[0]))
    for vec in vectors:
        sum_vectors = sum_vectors + vec
    return sum_vectors/len(vectors)

def infer_vector(entities, model, verbose = False):
    """Given a list of entities, returns the vector representing the document from which the entities 
    were extracted from, wrt a given W2V model.
    
    entities: list of entities, our way of representing a single document.
    model: w2v model.
    
    Returns (vector) representation of given document.
    """
    unknown_words = 0
    # get word vector of each entity; ignores word if the model does not know it
    entities_vecs = []
    for e in entities:
        try:
            # make sure to lower case each word!
            entities_vecs.append(model[e.lower()])
        except:
            unknown_words += 1 # ignore unknown word
    if unknown_words > 1 and verbose:
        print("Number of unknown words: ", unknown_words, ", number of known words:", (len(entities)-unknown_words))
    return mean_of_vectors(entities_vecs)
    