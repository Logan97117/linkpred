import subprocess

    
#count the number of k cliques
def cliques(path,k):
    
    a = int(subprocess.check_output('counts ' + str(k) + ' '+ path,shell = True))
    
    return a

        
