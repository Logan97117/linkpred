#libraries
import networkx as nx
import subprocess
import count_cliques
#subprocess.call('gcc clique_counter.c -O9 -o counts',shell = True)
subprocess.call('gcc clique_counter.c -O9 -o counts',shell = True)
    

#global 2nd order clustering coefficient
def global_coef_2(graph_object,path):
    k_3 = count_cliques.cliques(path,3)
    w_2 = 0
    for node in graph_object.nodes():
        k_2_node = 0
        for t in graph_object.edges():
            if(node in t):
                k_2_node = k_2_node + 1 
        w_2 = w_2 + (k_2_node * (graph_object.degree(node)-1))        
    
    return (6 * k_3)/(w_2)            
        
    
#traditional local clustering coefficient of a node
def local_coef_2(graph_object, node):
    return nx.clustering(graph_object,node)
    
#3rd order global clustering coefficient
def global_coef_3(graph_object,path):
    k_4 = count_cliques.cliques(path,4)
    w_3 = 0
    for node in graph_object.nodes():
        g_sub = graph_object.subgraph(nx.neighbors(graph_object,node))
        k_3_node = len(g_sub.edges())
        w_3_node = k_3_node * (graph_object.degree(node) - 2)
        w_3 = w_3 + w_3_node
        
        
        
        
        
    return (12 * k_4)/(w_3) 

    
#3rd order local clustering coefficient of a node
#defined for nodes which are centres of atleast one 3 wedge
def local_coef_3(graph_object,node,path):
    if(graph_object.degree(node)>=3):
        g_sub = graph_object.subgraph(nx.neighbors(graph_object,node))
        nx.write_edgelist(g_sub,'edgelist2.txt',data = False)
        k_4_node = count_cliques.cliques('edgelist2.txt',3)
        k_3_node = len(g_sub.edges())
        
        if(k_3_node==0):
            return 0
        else:    
            return (3 * k_4_node)/((graph_object.degree(node) - 2)* k_3_node)
        
    else:
        return 0
            
          
#4th order global clustering coefficient
def global_coef_4(graph_object,path):
    k_5 = count_cliques.cliques(path,5)
    w_4 = 0
    for node in graph_object.nodes():
        g_sub = graph_object.subgraph(nx.neighbors(graph_object,node))
        nx.write_edgelist(g_sub,'edgelist2.txt',data = False)
        k_4_node = count_cliques.cliques('edgelist2.txt',3)
        w_4_node = k_4_node * (graph_object.degree(node) - 3)
        w_4 = w_4 + w_4_node
        
    return (20 * k_5)/(w_4)            
    
#4th order local clustering coefficient
#defined for nodes which are centres of atleast one 4 wedge
def local_coef_4(graph_object,node,path):
    if(graph_object.degree(node)>=4):
        g_sub = graph_object.subgraph(nx.neighbors(graph_object,node))
        nx.write_edgelist(g_sub,'edgelist2.txt',data = False)
        k_5_node = count_cliques.cliques('edgelist2.txt',4)
        k_4_node = count_cliques.cliques('edgelist2.txt',3)
        
        if(k_4_node==0):
            return 0
        else:    
            return (4 * k_5_node)/((graph_object.degree(node) - 3)* k_4_node)
        
        
    else:
        return 0

    
