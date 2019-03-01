import networkx as nx

G = nx.read_edgelist('edgelist.txt',nodetype = int)


cliques = nx.cliques_containing_node(G)
for key,c in cliques.items():
    c = sorted(c, key = lambda x : -len(x))
    cliques[key] = c
cnt_3 = 0
for lst in cliques.values():
    if(len(lst)==3):
        cnt_3+=1
        
        
cliques.values()        