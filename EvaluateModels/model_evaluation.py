


def compute_clustering_accuracy(clusters_res, clusters_exp, verbose = False, alg = 'avg'):
    """ 
        No duplicates allowed.  No single-cluster elements allowed in the clusters_exp composition.
        
        cluster_res: list of clusters, result of the dbscan process; each cluster contains
        a variable number of docs, each document is represented by its TITLE. (e.g. [ ['doc_titleA', 'doc_titleB'], [..],...])
        
        cluster_exp: list of expected clusters, given by the evaluation set; we're expecting 
        this list to have the same format as above (list of titles).
        
        alg: which algorithm to use when computing 'best' result for Precision and Recall; max 
        keeps the best P,R for each cluster, min the worse(pessimistic), avg by default.
        
        The accuracy measure returned is known as 'recall' (true positives/sum of true pos. and false negatives)
        Given these two args, we return the accuracy percentage, calculated using the (soft) test rules:
            - a cluster in clusters_exp gets a 100% accuracy if all its documents are grouped in the same 
              cluster (these 2 cluster does NOT have to be exactly equal).
            - a cluster in clusters_exp gets a 1-99% accuracy if a sub-set of the doc is correctly
              grouped together (we take the biggest sub-set among different ones).
            - a cluster in clusters_exp gets a 0% accuracy rate if the greatest sub-set of correctly 
              grouped docs has a length of 1 (0, in case all docs are classified as noise).
       
        Return the overall accuracy as the average of the accuracy over each cluster.
        
        The second returned measure is known as 'precision':
        it is computed by calculating the number of correctly grouped docs (true positives),
        divided by the number of the elements in the cluster they're grouped in (sum of true positives and false positives).
        
        It's 1.0 (100% correct) if the expected cluster and the result cluster both contain exactly the same elements:
        this way we can recognize models that tend to have a great 'recall' score, by grouping all elements together in the same 
        cluster, since they will obtain a very poor 'precision score', by not recognizing different clusters. 
        
        Return precision score average (over every cluster), and recall averaged likewise.
    """
   

    p_results = []
    r_results = []
    # go through each expected cluster  
    for doc_titles in clusters_exp:
        occurences = []
        # check if the docs inside this cluster are grouped together
        if verbose: print("##Searching for docs", doc_titles, '##')
       
        # check in cluster 1, then in cluster 2..
        for index, cluster in enumerate(clusters_res):
            if verbose: print("==Searching inside cluster", cluster,"==")
            correct = 0
            for title in doc_titles:
                if title in cluster:
                    if verbose: print(title[:10],'.. is in cluster!')
                    correct += 1
            # save how many docs we found in this cluster, and the cluster index
            # this is the number of 'true positives' (correctly grouped)
            occurences.append((correct, index))
        # here we have all the occurences of docs computed in this iteration
        # we compute the Precision and Recall values, then store the best/worst/avg depending on arg
        ps, rs = [], []
        for (true_pos, c_index) in occurences:
            if true_pos > 0: # you could ignore clusters with only 1 document classified too
                ps.append(true_pos/len(clusters_res[c_index]) * 100) 
                rs.append(true_pos/len(doc_titles) * 100)
        if len(ps) > 0: # cluster was classified as noise otherwise
            if alg=='max':
                # keep the best P and the best R
                p_results.append(max(ps))
                r_results.append(max(rs))
            elif alg=='min':
                p_results.append(min(ps))
                r_results.append(min(rs))
            else:
                p_results.append(sum(ps)/len(ps))
                r_results.append(sum(rs)/len(rs))
            
     # print the result list [ (most_occurences, cluster in which they occur the most)..]
    #print("Results occurences(correct guess, cluster index): ", results_occurences)
    
    
    ## PRECISION
    print("Accuracy (Precision) over each cluster: ", p_results)

    ## RECALL
    print("Accuracy (Recall) over each cluster: ", r_results)
    
    
    # compute the mean of percentages as final accuracy of the model over the test set (in terms of clustering)

    precision_avg = sum(p_results)/len(p_results)

    recall_avg = sum(r_results)/len(r_results)
    
    
    # return PRECISION, RECALL
    return precision_avg, recall_avg



