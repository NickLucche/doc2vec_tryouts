


def compute_clustering_accuracy(clusters_res, clusters_exp, verbose = False):
    """ 
        No duplicates allowed.  No single-cluster elements allowed in the clusters_exp composition.
        
        cluster_res: list of clusters, result of the dbscan process; each cluster contains
        a variable number of docs, each document is represented by its TITLE. (e.g. [ ['doc_titleA', 'doc_titleB'], [..],...])
        
        cluster_exp: list of expected clusters, given by the evaluation set; we're expecting 
        this list to have the same format as above (list of titles).
        
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
   

    results_occurences = []
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
            occurences.append((correct, index))
        # here we have all the occurences of docs computed in this iteration
        # we store the best result:
        # order list by most appearences and keep the last pair
        occurences.sort(key = lambda tup: tup[0])
        results_occurences.append(occurences[-1])
     # print the result list [ (most_occurences, cluster in which they occur the most)..]
    print("Results occurences(correct guess, cluster index): ", results_occurences)
    
    # compute percentages
    correct = [c for (c, index) in results_occurences]
    percentages_recall = []
    percentages_precision = []
    
    ## PRECISION
    for (corrects, cindex) in results_occurences:
        # if less than 2 docs were correctly classified, the 'answer' is considered not correct
        if corrects < 2:
            percentages_precision.append(0)
        else:
            print("Corrects/Length cluster they're in", corrects, len(clusters_res[cindex]))
            percentages_precision.append(corrects * 100 / len(clusters_res[cindex]))
    print("Accuracy (Precision) over each cluster: ", percentages_precision)

    ## RECALL
    for i, docs_titles in enumerate(clusters_exp):
        # if less than 2 docs were correctly classified, the 'answer' is considered not correct
        if correct[i] < 2:
            percentages_recall.append(0)
        else:
            percentages_recall.append(correct[i] * 100 / len(docs_titles))
    print("Accuracy (Recall) over each cluster: ", percentages_recall)
    
    
    # compute the mean of percentages as final accuracy of the model over the test set (in terms of clustering)
    p_sum = 0
    for p in percentages_precision:
        p_sum += p
    precision_avg = p_sum/len(percentages_precision)
    
    p_sum = 0
    for p in percentages_recall:
        p_sum += p
    recall_avg = p_sum/len(percentages_recall)
    
    
    # return PRECISION, RECALL
    return precision_avg, recall_avg



