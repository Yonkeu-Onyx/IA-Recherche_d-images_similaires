import numpy as np 
#from scipy.stats import distance
from scipy.spatial.distance import canberra

def manhattan(v1,v2):
    v1=np.array(v1).astype('float')
    v2=np.array(v2).astype('float')
    dist=np.sum(np.abs(v1-v2))
    return dist

def euclidienne(v1,v2):
    v1=np.array(v1).astype('float')
    v2=np.array(v2).astype('float')
    dist=np.sqrt(np.sum(v1-v2)**2)
    return dist
def chebyshev(v1,v2):
    v1=np.array(v1).astype('float')
    v2=np.array(v2).astype('float')
    dist=np.max(np.abs(v1-v2))
    return dist

def canberra_dist(v1,v2):
    v1=[float(x) for x in v1]
    v2=[float(x) for x in v2]
    
    return canberra(v1,v2)
    
def Recherche_images_similaire(bdd_signatures,carac_requete,Distances,K):
    list_img_similaire=[]
    for instance in bdd_signatures:
        carac,label,img_chemin=instance[:-2],instance[-2],instance[-1]
        if Distances=='Manhattan':
            dist=manhattan(carac,carac_requete)
        if Distances=='Euclidienne':
            dist=euclidienne(carac,carac_requete)
        if Distances=='Chebychev':
            dist=chebyshev(carac,carac_requete)
        if Distances=='Canberra':
            dist=canberra_dist(carac,carac_requete)
        list_img_similaire.append((dist,label,img_chemin))
    list_img_similaire.sort(key=lambda x:x[0])
    return list_img_similaire[:K]