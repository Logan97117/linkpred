#Link prediction based on similarity of nodes
#training and test split is carried out on 90% and 10% of the dataset
#import libraries
import networkx as nx
import similarity_indices as si
#import higher_order_clustering_coefficients as hocc
import metrics
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import time
import random
import numpy as np


#Graph object and cliques 
G = nx.read_edgelist('C:\\Users\\dell i\\Desktop\\Research\\Social network analysis\\datasets\\dolphins\\edgelist.txt',nodetype = int)
           




#Evaluation
if(len(G.edges())>=1000):
    l=100
    L = [10,20,30,40,50,60,70,80,90,100]
    
else:
    l=20 
    L = [2,4,6,8,10,12,14,16,18,20]
   
    



def evaluationResults(graph_object,algorithm,T):
    start_time = time.time()
    prec = 0
    auc = 0
    ranking_score = 0
    iter = 0
    for i in range(T):
        G2 = graph_object.copy()
        non_existing = list(nx.non_edges(G2))
        random.seed(100)
        E_p = random.sample(list(G2.edges()),int(0.1*len(G2.edges())))
        for link in E_p:
                G2.remove_edge(link[0],link[1])
        missing_links_score = getattr(si,algorithm)(G2,E_p)
        non_existing_links_score = getattr(si,algorithm)(G2,non_existing)
        #missing_links_score = getattr(si,algorithm)(G2,E_p,'edgelist.txt')
        #non_existing_links_score = getattr(si,algorithm)(G2,non_existing,'edgelist.txt')
        non_observed_links_score = missing_links_score + non_existing_links_score
        prec = prec + metrics.precision(non_observed_links_score,missing_links_score,l)
        auc = auc + metrics.AUC(non_existing_links_score,missing_links_score)
        ranking_score = ranking_score + metrics.RS(non_observed_links_score,missing_links_score)
        iter = iter + 1
        print("Iteration: " + str(iter))
        
    end_time = time.time()
    print(str(T)+ " fold Precision: " + str(prec/10))
    print(str(T) + " fold AUC: " + str(auc/10))
    print("Mean Ranking score of the algorithm: " + str(ranking_score/10))
    print("Time taken for evaluation: " + str(end_time - start_time) + " seconds")

   
def precision_curve(graph_object,T):
    algorithms = ['CAR','CCLP_2']
    algo_precision_matrix = []
    for algo in algorithms:
        print("Algorithm running:" + algo)
        precision = []
        for l_value in L:
            print("Computing for threshold:" + str(l_value))
            prec = 0
            for i in range(T):
                G2 = graph_object.copy()
                non_existing = list(nx.non_edges(G2))
                random.seed(100)
                E_p = random.sample(list(G2.edges()),int(0.1*len(G2.edges())))
                for link in E_p:
                        G2.remove_edge(link[0],link[1])
                missing_links_score = getattr(si,algo)(G2,E_p)
                non_existing_links_score = getattr(si,algo)(G2,non_existing)
                #missing_links_score = getattr(si,algo)(G2,E_p,'edgelist.txt')
                #non_existing_links_score = getattr(si,algo)(G2,non_existing,'edgelist.txt')
                non_observed_links_score = missing_links_score + non_existing_links_score
                prec = prec + metrics.precision(non_observed_links_score,missing_links_score,l_value)
            precision.append(prec/10)
            
        algo_precision_matrix.append(precision)
         
    for i in range(len(algorithms)):
        plt.plot(L,algo_precision_matrix[i])
        plt.scatter(L,algo_precision_matrix[i],label = algorithms[i])
        plt.xticks(L)
        plt.yticks(np.linspace(start = 0.01,stop = 1.0,num = 10,dtype=float))
        plt.xlabel('L values')
        plt.ylabel('10 fold precision values')
        plt.legend()
    plt.show()    
        

#AUC as a function of length of added or removed links from training set for a particular algorithm
def AUC_under_noise(graph_object,algorithm,num,func):
    if(func=='added'):
        G2 = graph_object.copy()
        non_existing_links = list(nx.non_edges(G2))
        random.seed(100)
        #probe set and training set from existing links
        E_p = random.sample(list(G2.edges()),int(0.1*len(G2.edges())))
        E_t = [edge for edge in list(G2.edges()) if edge not in E_p]
        for link in E_p:
            G2.remove_edge(link[0],link[1])
        len_added = num
        edges_added = random.sample(non_existing_links,len_added)
        G2.add_edges_from(edges_added)
        non_existing_links = [edge for edge in non_existing_links if edge not in edges_added]
        missing_links_score = getattr(si,algorithm)(G2,E_p)
        non_existing_links_score = getattr(si,algorithm)(G2,non_existing_links)

        return metrics.AUC(non_existing_links_score,missing_links_score)

    if(func=='None' and num==0):
        G2 = graph_object.copy()
        non_existing = list(nx.non_edges(G2))
        random.seed(100)
        E_p = random.sample(list(G2.edges()),int(0.1*len(G2.edges())))
        for link in E_p:
            G2.remove_edge(link[0],link[1])
        missing_links_score = getattr(si,algorithm)(G2,E_p)
        non_existing_links_score = getattr(si,algorithm)(G2,non_existing)
        return metrics.AUC(non_existing_links_score,missing_links_score)

    if(func=='remove'):
        G2 = graph_object.copy()
        #initially non existing links
        non_existing_links = list(nx.non_edges(G2))
        random.seed(100)
        #probe set and training set from existing links
        E_p = random.sample(list(G2.edges()),int(0.1*len(G2.edges())))
        E_t = [edge for edge in list(G2.edges()) if edge not in E_p]
        for link in E_p: #probe set removed
            G2.remove_edge(link[0],link[1])
        len_removal = num
        edges_removed = random.sample(E_t,len_removal) # ratio less than zero, edges removed from E_t and goes to non_existing, probe is constant
        for edge in edges_removed:
            G2.remove_edge(edge[0],edge[1])
        non_existing_links = non_existing_links + edges_removed
        missing_links_score = getattr(si,algorithm)(G2,E_p)
        non_existing_links_score = getattr(si,algorithm)(G2,non_existing_links)
        return metrics.AUC(non_existing_links_score,missing_links_score)
        

#k fold robustness of a particular algorithm under a particular ratio
def k_fold_robustness(graph_object,algorithm,ratio,T):
    robustness = 0
    for i in range(T):
        if(ratio>0):
            sum_auc = 0
            train_length = int(0.9 * len(list(graph_object.edges())))
            test_length = len(list(graph_object.edges())) - train_length
            L = int(ratio * train_length)
            for num in range(L+1):
                sum_auc = sum_auc + AUC_under_noise(graph_object,algorithm,num,'added')
            robustness = robustness + ((1/L) * (sum_auc/AUC_under_noise(graph_object,algorithm,0,'None')))
        if(ratio==0):
            robustness = robustness + 1        

        if(ratio<0):
            sum_auc = 0
            train_length = int(0.9 * len(list(graph_object.edges())))
            test_length = len(list(graph_object.edges())) - train_length
            L = int(abs(ratio * train_length))
            for num in range(L+1):
                sum_auc = sum_auc + AUC_under_noise(graph_object,algorithm,num,'remove')
            robustness = robustness + ((1/L) * (sum_auc/AUC_under_noise(graph_object,algorithm,0,'None')))

    return (robustness/T)             


#variation of AUC and robustness by adding or removing links with different ratio for different algorithms        
def AUC_robustness_variation_under_noise(graph_object): 
    algorithms = ['CN','Jaccard','RA','katz_similarity']
    auc_matrix = []
    robustness_matrix = []
    ratio = [-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]
    for algo in algorithms:
        auc_values = []
        robustness_values = []
        for r in ratio:
            if(r<0):
                train_length = int(0.9 * len(list(graph_object.edges())))
                test_length = len(graph_object.edges()) - train_length
                len_removal = int(abs(r * train_length))
                auc_values.append(AUC_under_noise(graph_object,algo,len_removal,'remove'))
                robustness_values.append(k_fold_robustness(graph_object,algo,r,1))
            if(r==0):
                G2 = graph_object.copy()
                non_existing = list(nx.non_edges(G2))
                random.seed(100)
                E_p = random.sample(list(G2.edges()),int(0.1*len(G2.edges())))
                for link in E_p:
                        G2.remove_edge(link[0],link[1])
                missing_links_score = getattr(si,algo)(G2,E_p)
                non_existing_links_score = getattr(si,algo)(G2,non_existing)
                auc_values.append(metrics.AUC(non_existing_links_score,missing_links_score))
                robustness_values.append(k_fold_robustness(graph_object,algo,r,1))
            if(r>0): # random links are to be added from inital non existing links to the training set, means add edges in graph, keeping probe set constant
                train_length = int(0.9 * len(list(graph_object.edges())))
                test_length = len(graph_object.edges()) - train_length
                len_added = int(r * train_length)
                auc_values.append(AUC_under_noise(graph_object,algo,len_added,'added'))
                robustness_values.append(k_fold_robustness(graph_object,algo,r,1))
        
        auc_matrix.append(auc_values)
        robustness_matrix.append(robustness_values)
    max_value_auc,max_value_rob = max(list(set().union(*auc_matrix))),max(list(set().union(*robustness_matrix)))
    min_value_auc,min_value_rob = min(list(set().union(*auc_matrix))),min(list(set().union(*robustness_matrix)))
    for i in range(len(algorithms)):
        plt.plot(ratio,auc_matrix[i],label = algorithms[i])
        plt.scatter(ratio,auc_matrix[i])
        plt.xticks(ratio)
        plt.yticks(np.linspace(start = max_value_auc,stop = min_value_auc,num = 10,dtype=float))
        plt.xlabel('Ratio of noisy links(Sign tells whether added or removed)')
        plt.ylabel('AUC values')
        plt.legend()    
    plt.show()

    for i in range(len(algorithms)):
        plt.plot(ratio,robustness_matrix[i],label = algorithms[i])
        plt.scatter(ratio,robustness_matrix[i])
        plt.xticks(ratio)
        plt.yticks(np.linspace(start = max_value_rob,stop = min_value_rob,num = 10,dtype=float))
        plt.xlabel('Ratio of noisy links(Sign shows wheher added or removed)')
        plt.ylabel('AUC values')
        plt.legend()

    plt.show()
      



    



                


                    



   
#evaluationResults(G,'katz_similarity',10)
#precision_curve(G,10)
AUC_robustness_variation_under_noise(G)

        
    












    
    
    

    
    







        
    


























