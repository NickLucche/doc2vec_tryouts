#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing methods used to visualize results in plots.

@author: NickLucche
"""
#TODO: improve description
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
import random
import numpy as np
import sklearn.metrics.pairwise as sk # for cosine_distance

def plot_clusters(clusters):
    """
        Clusters: list of clusters, each cluster containing a document representation (its title, url, id..)
        
        This method will plot the given clusters, in a non-meaningful way in terms of clusters
        position in space, but rather focusing on clearly distinguishing clusters and 
        on showing their content.
    """
    # using my api-key
    tls.set_credentials_file(username='D4nt3', api_key='FdMB4O6qCfciGDOnLvdQ')
    
    
    
    traces = []
    c_colors = ['blue', 'red', 'yellow', 'green', 'pink', 'grey', 'black', 'magenta']
    
    # each trace will represent a cluster; the more documents a cluster contains,
    # the bigger the point visualized will be.
    # Hovering on point will pop-up a list showing cluster content.
    for i, cluster in enumerate(clusters):
        # assign a color to each cluster
        if i == 0:
            x, y = 4, 4
        else:
            x , y = random.randint((-1)**(i), (3)*(i)), random.randint((-1)**(i), (3)*(i))
            
        cluster_size = len(cluster)
        
        color = i
        text = ''
        for word in cluster:
            text = text+','+word
            
        trace0 = go.Scatter(
            x = [x], 
            y = [y],
            mode = 'markers',
            name = 'cluster'+str(i),
            marker = dict(
                size = 6 * cluster_size/2,
                color = color,
                colorscale = 'Viridis'
            ),
            text = text
        )
        traces.append(trace0)
    
    
    data = traces 
    layout = dict(title = 'Clusters visual representation',
                hovermode= 'closest',
                xaxis= dict(
                    title= 'x',
                    ticklen= 5,
                    gridwidth= 2,
                ),
                yaxis=dict(
                    title= 'y',
                    ticklen= 5,
                    gridwidth= 2,
                ),
                showlegend = False
            )
        
    fig = dict(data = data, layout = layout)
    #py.iplot(fig, filename= filename)
    # or return fig
    return fig 

def getDocTitleFromUrl(docs, clustered_urls):
    clusters_t = []
    for cluster in clustered_urls:
            cluster_titles = []
            for url in cluster:
                for doc in docs:
                    if doc['url'] == url:
                        cluster_titles.append(doc['title'])
            clusters_t.append(cluster_titles)
    return clusters_t  

def choose_eps(min_count, docs_vecs, doc_titles):
    """Plot the graph and choose best eps based on the composition of our data: 
    to do so, we will need to compute the distances between every point in the data-space, and its
    2nd/3rd closest neighbour (based on 'min_count'). 
    Take the eps corresponding to a great change in the derivative of the plotted function ('knee' or 'elbow' shape).
    
    Docs_vecs is the list of vectors we will analyze, each representing a document.
    
    min_count is the number of points needed to define a core point in DBSCAN.
    
    doc_titles is a matching list (wrt to docs_vecs), containing the titles of each doc.
    
    Returns a list of tuples (doc_title, distance from k-th neighbour), ORDERED by distance (ascendantly).
    """
    
    # first thing to do: compute the matrix of all pairwise elements distances
    # warning: this code is not optimized
    dist_matrix = get_pairwise_distances_matrix(docs_vecs)
    
    
    list_ = [] 
    # for each document vec, only keep the DISTANCE from k-th closest document
    for j, doc_distances in enumerate(dist_matrix):
        # get a row of the matrix (vector of distances for doc_j)
        
        # discard the distance between a doc and itself
        doc_distances = np.delete(doc_distances, j)
        for i in range(0, min_count-1):
            # get the closest doc to it and discard it, we only need the k-th closest doc.
            doc_distances = np.delete(doc_distances, np.argmin(doc_distances))
        # now create the pair: (doc_name, distance from k-th neighbour)
        list_.append((doc_titles[j], np.amin(doc_distances)))
        
    # sort the list by the second parameter (distance)
    list_.sort(key=lambda tup: tup[1])  # sorts in place
    return list_


# TODO: add possibility of passing metric to use as parameters
def get_pairwise_distances_matrix(docs):
    """"
        docs: list of documents, each represented as a vector.
        
        Returns the pairwise distances matrix between documents. 
    
        Metric used to compute the distance is cosine_distance -by default-.
    """
    # initialize distance matrix
    n = len(docs)
    distances_m = np.zeros((n, n))
    
    # compute the distance betweem each vector (doc)
    # this is all but efficient at the moment, okay for a debug version.
    for i, doc1 in enumerate(docs):
        for j, doc2 in enumerate(docs):
            distances_m[i, j] = sk.cosine_distances([doc1], [doc2])
    return distances_m

def mean_of_vectors(vectors):
    """given a list of vectors, return the simplest mean of vectors."""
    
    sum_vectors = np.zeros(np.shape(vectors[0]))
    for vec in vectors:
        sum_vectors = sum_vectors + vec
    return sum_vectors/len(vectors)

def infer_vector(entities, model):
    """Given a list of entities, returns the vector representing the documents from which the entities 
    were extracted from, wrt a given W2V model.
    
    entities: list of entities, our way of representing a document.
    model: w2v model.
    """
    
    # get word vector of each entity; ignores word if the model does not know it
    entities_vecs = []
    for e in entities:
        try:
            entities_vecs.append(model[e])
        except:
            None # ignore unknown word
    
    return mean_of_vectors(entities_vecs)
    

def visualize_eps_graph(title_dist_tuples):
    
    trace = go.Scatter(
        x =[x for (x, y) in title_dist_tuples],  # list of x
        y = [y for (x, y) in title_dist_tuples],
        mode = 'lines',
        name = 'lines'
    )
    
    data = [trace]
    # we need to return the data-trace, cause plotly won't work with module calls
    return data
    
    
    
def lower_case_list(list_):
    """ Given a list, returns an equal list, containing lower-cased elements of the original list."""
    return [word.lower() for word in list_]

def delete_duplicates_from_list(list_):
    """ Given a list, returns the same list without duplicates. """
    index_list = []
    for i, element in enumerate(list_):
        try:
            index = list_.index(i+1, len(list_), element)
            index_list.append(index)
        except:
            None
    return [el for j, el in enumerate(list_) if not(j in index_list)]
