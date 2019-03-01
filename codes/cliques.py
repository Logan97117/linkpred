import networkx as nx
import numpy as np

G = nx.read_edgelist('edgelist.txt',nodetype = int)



    

def _kcliques(A,k):
    
    G2 = nx.from_numpy_matrix(A)
    ad_list = []
    for node,d in G2.adjacency():
        ad_list.append(list(d.keys()))
    C = []
    labels = k* np.ones(len(G2.nodes()),dtype = int)
    counts = np.zeros(len(G2.nodes()),dtype = int) 
    
    #get neighbors of a node within given label
    def neighbors(node,label):
        index = 0
        for nbr in ad_list[node]:
            nbr_label = labels[nbr]
            if(nbr_label==label):
                index = index + 1
            if(nbr_label>label):
                break
        return ad_list[node][0:index]     
            
    
            
    # Get the number of neighbors of a node with a given label
    def num_neighbors(node,label):
        num = 0
        for nbr in ad_list[node]:
            nbr_label = labels[nbr]
            if(nbr_label==label):
                num = num + 1
            if(nbr_label>label):
                break
        return num 
        
    # Get the nodes in U in sorted order that need to be processed
    def nodes_to_process(U, label):
        sorted_nbrs = sorted([(num_neighbors(v, label), v) for v in U], reverse=True)
        return [x[1] for x in sorted_nbrs if x[0] > 0]
    
    
    # Adjust adjacency lists of neighbors of nodes in U to put neighbors with a
    # given label in the front.
    def move_front(U,label):
        for v in U:
            front_nbrs = []
            back_nbrs  = []
            for nbr in ad_list[v]:
                if(labels[nbr] == label):
                    front_nbrs.append(nbr)
                else:
                    back_nbrs.append(nbr)
                    
                
            num_front = len(front_nbrs)
            total = len(ad_list[v])
            ad_list[v][0:num_front-1] = front_nbrs
            ad_list[v][num_front:total] = back_nbrs
        
        
        
    # Adjust adjacency lists of neighbors of nodes in U to put node vi directly
    # after nodes with a given label.
    def move_after(U, vi, label):
        for v in U:
            front_nbrs = []
            for nbr in ad_list[v]:
                if(nbr == vi):
                    continue
                if(labels[nbr]<=label):
                    front_nbrs.append(nbr)
                else:
                    break
                    
                
            num_front = len(front_nbrs)
            ad_list[v][0:num_front-1] = front_nbrs
            ad_list[v][num_front] = vi
        
        
    #process current cliques
    def process_cliques(U):
        C_cnt = 0
        for v in U:
            v_cnt = 0
            for nbr in neighbors(v,2):
                if(nbr>v):
                    counts[nbr]+=1
                    v_cnt+=1
            counts[v]+=v_cnt    
            C_cnt+=v_cnt    
            
        counts[C]+=C_cnt
        
    
    #recursive algorithm of Chiba and Nishizeki
    def rcliques(U,r):
        #check base case
        if(r==2):
            process_cliques(U)
            return
            
        for (ind,v) in enumerate(nodes_to_process(U,r)):
            #if(r==k):
                #print("%d of %d (%d-cliques) \r", ind, len(U), r)
                #if(ind == len(U)):
                    #print("");
            
            
            
            #get neighbors of v
            Up = neighbors(v,r)
            labels[Up] = r-1
            move_front(Up,r-1)
            
            #recurse on neighbors of v
            C.append(v)
            rcliques(Up,r-1)
            C.pop()
            
            #restore neighborhood
            labels[Up] = r
    
            #eliminate v
            labels[v] = r+1
            move_after(Up,v,r)
            
        
    u1 = []
    c = np.sum(A,axis = 1)
    for i in range(len(c)):
        if(c[i]>=k):
            u1.append(i)
    rcliques(u1,k)  
    return counts


def kcliques(graph_object,k):
    A = nx.adjacency_matrix(graph_object).todense()
    clique_counts = np.zeros(len(graph_object.nodes()),dtype = int)
    d = list(nx.core_number(graph_object).values())
    ind = []
    for i in range(len(d)):
        if(d[i]>=k-1):
            ind.append(i)

        
    ind = np.array(ind)
    counts = _kcliques(A[ind[:,None],ind],k)
    clique_counts[ind] = counts
    return clique_counts
    
    
 
print(sum(kcliques(G,3)))

