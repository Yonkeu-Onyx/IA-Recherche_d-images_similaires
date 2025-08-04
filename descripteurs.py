import cv2 
from skimage.feature import graycomatrix,graycoprops
from BiT import bio_taxo
from mahotas.features import haralick
import numpy as np



# Descripteurs RGB---------------------

def glcm_RGB(image):
    # data=cv2.imread(chemin)
    # data = image
    data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    list_carac = []
    for i in range(3):
        canal=data[:,:,i]
        co_matrice=graycomatrix(canal,distances=[1],angles=[3*np.pi/2],symmetric=False,normed=True)
        contrast = graycoprops(co_matrice,'contrast')[0,0]
        dissimilarity = graycoprops(co_matrice,'dissimilarity')[0,0]
        homogeneity = graycoprops(co_matrice,'homogeneity')[0,0]
        correlation = graycoprops(co_matrice,'correlation')[0,0]
        energy = graycoprops(co_matrice,'energy')[0,0]
        ASM = graycoprops(co_matrice,'ASM')[0,0]
        
        features = [contrast,dissimilarity,homogeneity,correlation,energy,ASM]
        
        features = [float(x) for x in features]
        list_carac.extend(features)
    return list_carac


def haralick_feat_RGB(image):
    # data=cv2.imread(chemin)
    # data = image
    data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    list_carac=[]
    for i in range(3):
        canal=data[:,:,i]
        features=haralick(canal).mean(0).tolist()
        features=[float(x) for x in features]
        list_carac.extend(features)
    return list_carac

def bitdesc_feat_RGB(image):
    # data=cv2.imread(chemin)
    # data = image
    data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    list_carac=[]
    for i in range(3):
        canal=data[:,:,i]
        features=bio_taxo(canal)
        features=[float(x) for x in features]
        list_carac.extend(features)
    return list_carac

def concatenation_RGB(image):
    return glcm_RGB(image)+haralick_feat_RGB(image)+bitdesc_feat_RGB(image)