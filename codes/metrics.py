#libraries
from sklearn.metrics import accuracy_score
import networkx as nx
import numpy as np
from random import shuffle
from bisect import bisect_left


# precision values
def precision(non_observed_score,missing_links_score,L):
    threshold = sorted(non_observed_score,reverse = True)[L-1]
    true = np.ones(len(missing_links_score))
    pred = np.zeros(len(missing_links_score))
    for i in range(len(missing_links_score)):
        if(missing_links_score[i]>=threshold):
            pred[i] = 1
        else:
            pred[i] = 0
                
    return accuracy_score(true,pred) * (len(missing_links_score)/L) 
    
    
    
    
#Ranking score
def RS(non_observed_links_score,missing_links_score):
    non_observed_links_score = sorted(non_observed_links_score,reverse = True)
    ranking_scores_missing_links = np.empty(shape = len(missing_links_score)) #empty array of ranking score of probe set links
    for i in range(len(missing_links_score)):
            rank = non_observed_links_score.index(missing_links_score[i]) + 1
            ranking_scores_missing_links[i] = rank/len(non_observed_links_score)
    RS = 1/(len(missing_links_score) * np.sum(ranking_scores_missing_links))
    return RS


#AUC values
def AUC(non_existing_score,missing_links_score):
    N = min(len(non_existing_score),len(missing_links_score))
    n1 = n2 = 0
    shuffle(non_existing_score)
    shuffle(missing_links_score)
    for i in range(N):
        if(missing_links_score[i]>non_existing_score[i]):
            n1+=1
        if(missing_links_score[i]==non_existing_score[i]):
            n2+=1
            
            
    return (n1 + 0.5*n2)/N     



    
    

  
    
    
    
    
    
    
    












    
    
        
        

