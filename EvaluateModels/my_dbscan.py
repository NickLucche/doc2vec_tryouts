#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing utils used for clustering.
@author: NickLucche
"""

from sklearn.cluster import DBSCAN

# my function for performing dbscan and printing out cluster results
def perform_dbscan(eps = 0.4, min_samples = 4, metric = 'euclidean', algorithm = 'auto', data = None, verbose = True
                  , titles = None, urls = None, print_noise = True):
    """perform DBSCAN over given data, using given parametrs. Returns dbscan object and clusters dictionary."""
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(data)

    # labels will print out the number of the cluster each example belongs to;
    # -1 if the vector is considered noise (not belonging to any cluster)
    #print("Labels: ", db.labels_)

    # create data structure containing clusters
    clusters_to_ret = {label:[] for label in db.labels_ if label!=-1}
    
    for i, label in enumerate(db.labels_):
        if label != -1: #ignore noise points
            clusters_to_ret[label].append(urls[i])
        
    
    
    # only do this if you need to print out the result (messy for large number of docs)
    if verbose:
        print("##Clusters##")
        clusters = {label: [] for label in db.labels_ if label!=-1}
        noise = []
        for i, label in enumerate(db.labels_):
            if label != -1: 
                clusters[label].append(titles[i])
            else: # save noise points
                noise.append(titles[i])
                
        for label, list_ in clusters.items():
            print("Cluster: {}".format(list_))
        if print_noise:
            print("Noise: ", noise)

        print("DBSCAN finished.\n")
    return db, clusters_to_ret

def apply_dbscan(doc_vecs, titles, urls, subset_length, eps = 0.27, eps_increment = 0.1, n_iterations = 1, 
                 verbose = False, min_samples = 2):
    """
        This method performs a dbscan clustering and returns the resulting clusters,
        as a list of urls.
        Parameters are pretty self-explanatory; multiple iterations version doesn't yet implement
        a heuristic.
    """
    # TODO: improve description
    
    # subset of docs vectors 
    subset = doc_vecs[:subset_length]
    subset_titles = titles[:subset_length]
    sub_urls = urls[:subset_length]
    
    noise_bool = False
    # this will contain all clusters found, each one as a list, 
    # mantaining the order dbscan returned (first clusters will contain articles more related to each other)
    final_clusters = []
    # starting eps will be the sum of eps + eps_increment 
    for i in range(n_iterations):
        if i==2: 
            noise_bool = True
        eps = eps + eps_increment
       
        db, clusters = perform_dbscan(eps = eps, min_samples = min_samples, metric = 'cosine', algorithm = 'auto',
                            data = subset, verbose = verbose, titles = subset_titles, urls = sub_urls, print_noise = noise_bool)
    
        # TODO: ignore noise/'other' documents or return them?
        for label, list_ in clusters.items():
            final_clusters.append(list_)
            
        # let's try and find other clusters in the noise data, with higher eps
        subset = [subset[i] for i, label in enumerate(db.labels_) if label==-1]
        subset_titles = [subset_titles[i] for i, label in enumerate(db.labels_) if label==-1]
        sub_urls = [sub_urls[i] for i, label in enumerate(db.labels_) if label==-1]
        if subset is None:
            break
    
    if verbose:
        print("Number of cluster found: ", len(final_clusters))
        for i, cluster in enumerate(final_clusters):
            print("Length of cluster {0}: {1}".format(i, len(cluster)))
    # final clusters composition:
    #[[cluster0_urls], [cluster1_urls], ...]
    return final_clusters
