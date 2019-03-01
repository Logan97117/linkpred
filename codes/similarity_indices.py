#import libraires
import networkx as nx
from itertools import combinations
import higher_order_clustering_coefficients as hocc
import linkpred
import numpy as np



#PA
def PA(graph_object,list_node_tuple):
    pa_scores = []
    for t in nx.preferential_attachment(graph_object,list_node_tuple):
        pa_scores.append(t[2])
    return pa_scores    
        
#CN
def CN(graph_object,list_node_tuple):
    cn_score = []
    for t in list_node_tuple:
        cn_score.append(len(list(nx.common_neighbors(graph_object,t[0],t[1]))))
    return cn_score

#jaccard
def Jaccard(graph_object,list_node_tuple):
    jc_score = []
    for t in nx.jaccard_coefficient(graph_object,list_node_tuple):
        jc_score.append(t[2])
    return jc_score 

#AA
def AA(graph_object,list_node_tuple):
    aa_score = []
    for t in nx.adamic_adar_index(graph_object,list_node_tuple):
        aa_score.append(t[2])
    return aa_score 

#RA
def RA(graph_object,list_node_tuple):
    ra_score = []
    for t in nx.resource_allocation_index(graph_object,list_node_tuple):
        ra_score.append(t[2])
    return ra_score 

#CAR score
def CAR(graph_object,list_node_tuple):
    cn_count = CN(graph_object,list_node_tuple)
    lcl_count = []
    for t in list_node_tuple:
        lcl = 0
        for combi in combinations(nx.common_neighbors(graph_object,t[0],t[1]),2):
            if(graph_object.has_edge(combi[0],combi[1])):
               lcl+=1
        lcl_count.append(lcl)       

    car_scores = [a*b for a,b in zip(cn_count,lcl_count)]  
    return car_scores


#CCLP scores using 2nd order clustering coefficient
def CCLP_2(graph_object,list_node_tuple):
    cclp_scores = []
    for t in list_node_tuple:
        cclp = 0
        for n in list(nx.common_neighbors(graph_object,t[0],t[1])):
            cclp = cclp + nx.clustering(graph_object,n)
        cclp_scores.append(cclp)  
    return cclp_scores  
 
#CCLP_scores using 3rd order clustering coefficients
def CCLP_3(graph_object,list_node_tuple,path):
    cclp_scores = []
    for t in list_node_tuple:
        cclp = 0
        for n in list(nx.common_neighbors(graph_object,t[0],t[1])):
            cclp = cclp + hocc.local_coef_3(graph_object,n,path)
        cclp_scores.append(cclp)    
    return cclp_scores
    
#CCLP scores suing 4th order clustering coefficients    
def CCLP_4(graph_object,list_node_tuple,path):
    cclp_scores = []
    for t in list_node_tuple:
        cclp = 0
        for n in list(nx.common_neighbors(graph_object,t[0],t[1])):
            cclp = cclp + hocc.local_coef_4(graph_object,n,path)
        cclp_scores.append(cclp)    
    return cclp_scores
    
    
    
def sum_higher_order_cclp(graph_object,list_node_tuple,path):
    scores = []
    for t in list_node_tuple:
        sum_node_1 = hocc.local_coef_2(graph_object,t[0]) + hocc.local_coef_3(graph_object,t[0],path) + hocc.local_coef_4(graph_object,t[0],path)
        sum_node_2 = hocc.local_coef_2(graph_object,t[1]) + hocc.local_coef_3(graph_object,t[1],path) + hocc.local_coef_4(graph_object,t[1],path)
        total_sum = sum_node_1 + sum_node_2
        scores.append(total_sum)
    return scores
  

    
def simrank(graph_object,list_node_tuple):
    all_possible = list(combinations(graph_object.nodes(),2))
    excluded_pairs = [tuple for tuple in all_possible if tuple not in list_node_tuple]
    simrank = linkpred.predictors.SimRank(graph_object,excluded = excluded_pairs)
    results = simrank.predict()
    return list(results.values())

    
def rooted_pagerank(graph_object,list_node_tuple):
    scores = []
    for t in list_node_tuple:
        rooted_pagerank = linkpred.network.rooted_pagerank(graph_object,root = t[0])
        scores.append(rooted_pagerank.get(t[1]))
        
    return scores    
    
def all_cliques(graph_object,list_node_tuple):
    scores = []
    for t in list_node_tuple:
        sum_cliques = 0
        for node in nx.common_neighbors(graph_object,t[0],t[1]):
            sum_cliques = sum_cliques + len(nx.cliques_containing_node(graph_object,node))
        scores.append(sum_cliques)
        
    return scores    

#has O(n^3) complexity, not optimized
def katz_similarity(graph_object,list_node_tuple,beta = 0.1):
    scores = []
    adj_matrix = nx.adjacency_matrix(graph_object).todense()
    similarity_matrix = np.linalg.inv(np.identity(n=adj_matrix.shape[0],dtype=float) - (beta*adj_matrix)) - np.identity(adj_matrix.shape[0])
    for t in list_node_tuple:
        scores.append(similarity_matrix[t[0],t[1]])
    return scores



#Weak clique structure and friend recommendation model(FR) index, below are the subroutines for the FR score and various variations of FR score
def PWCS_probabilities(graph_object):
    






    
    
    


    
    
    
    
    
    
    
    



    
    
    
    
