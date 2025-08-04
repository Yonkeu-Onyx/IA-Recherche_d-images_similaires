from descripteurs import glcm_RGB,haralick_feat_RGB,bitdesc_feat_RGB,concatenation_RGB
import os
import cv2
import numpy as np
 
 
def extraction_signatures_glcm_RGB(chemin_repertoire):
    list_carac=[]
    for root,_,files in os.walk(chemin_repertoire):
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                relative_path=os.path.relpath(os.path.join(root,file),chemin_repertoire)
                #print(relative_path)
                #print(os.path.join(root,file))
                chemin=os.path.join(root,file)
                caracteristiques=glcm_RGB(chemin)
                class_name=os.path.dirname(relative_path)
                caracteristiques=caracteristiques+[class_name,relative_path]
                print(caracteristiques)
                list_carac.append(caracteristiques)
        signatures=np.array(list_carac)
        np.save('Signatures_GLCM_RGB.npy',signatures)
 
def extraction_signatures_Concat_RGB(chemin_repertoire):
    list_carac=[]
    for root,_,files in os.walk(chemin_repertoire):
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                relative_path=os.path.relpath(os.path.join(root,file),chemin_repertoire)
                #print(relative_path)
                #print(os.path.join(root,file))
                chemin=os.path.join(root,file)
                caracteristiques=concatenation_RGB(chemin)
                class_name=os.path.dirname(relative_path)
                caracteristiques=caracteristiques+[class_name,relative_path]
                print(caracteristiques)
                list_carac.append(caracteristiques)
        signatures=np.array(list_carac)
        np.save('Signatures_Concat_RGB.npy',signatures)
        
def extraction_signatures_haralick_RGB(chemin_repertoire):
    list_carac=[]
    for root,_,files in os.walk(chemin_repertoire):
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                relative_path=os.path.relpath(os.path.join(root,file),chemin_repertoire)
                #print(relative_path)
                #print(os.path.join(root,file))
                chemin=os.path.join(root,file)
                caracteristiques=haralick_feat_RGB(chemin)
                class_name=os.path.dirname(relative_path)
                caracteristiques=caracteristiques+[class_name,relative_path]
                print(caracteristiques)
                list_carac.append(caracteristiques)
        signatures=np.array(list_carac)
        np.save('Signatures_haralick_RGB.npy',signatures)
    
def extraction_signatures_bitdesc_RGB(chemin_repertoire):
    list_carac=[]
    for root,_,files in os.walk(chemin_repertoire):
        for file in files:
            if file.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                relative_path=os.path.relpath(os.path.join(root,file),chemin_repertoire)
                #print(relative_path)
                #print(os.path.join(root,file))
                chemin=os.path.join(root,file)
                caracteristiques=bitdesc_feat_RGB(chemin)
                class_name=os.path.dirname(relative_path)
                caracteristiques=caracteristiques+[class_name,relative_path]
                print(caracteristiques)
                list_carac.append(caracteristiques)
        signatures=np.array(list_carac)
        np.save('Signatures_bitdesc_RGB.npy',signatures)
 
 
if __name__=='__main__':
    extraction_signatures_glcm_RGB('./dataset/')
    extraction_signatures_Concat_RGB('./dataset/')
    extraction_signatures_haralick_RGB('./dataset/')
    extraction_signatures_bitdesc_RGB('./dataset/')
 
 
 
 
 
 